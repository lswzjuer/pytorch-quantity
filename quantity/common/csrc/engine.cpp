/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "engine.h"

#include <iostream>
#include <fstream>

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include "plugins/DecodePlugin.h"
#include "plugins/NMSPlugin.h"
#include "calibrator.h"

using namespace nvinfer1;
using namespace nvonnxparser;

namespace roadtensor {

class Logger : public ILogger {
public:
    Logger(bool verbose)
        : _verbose(verbose) {
    }

    void log(Severity severity, const char *msg) override {
        if (_verbose || (severity != Severity::kINFO))
            cout << msg << endl;
    }

private:
   bool _verbose{false};
};

void Engine::_load(const string &path) {
    ifstream file(path, ios::in | ios::binary);
    file.seekg (0, file.end);
    size_t size = file.tellg();
    file.seekg (0, file.beg);

    char *buffer = new char[size];
    file.read(buffer, size);
    file.close();

    _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);

    delete[] buffer;
}

void Engine::_prepare() {
    _context = _engine->createExecutionContext();
    cudaStreamCreate(&_stream);
}

Engine::Engine(const string &path, bool verbose) {
    Logger logger(verbose);
    _runtime = createInferRuntime(logger);
    _load(path);
    _prepare();
}

Engine::~Engine() {
    if (_stream) cudaStreamDestroy(_stream);
    if (_context) _context->destroy();
    if (_engine) _engine->destroy();
    if (_runtime) _runtime->destroy();
}

Engine::Engine(const char *onnx_model, size_t onnx_size, size_t batch, string precision,
    float score_thresh, int top_n, const vector<vector<float>>& anchors,
    float nms_thresh, int detections_per_im, const vector<string>& calibration_images, 
    string model_name, string calibration_table, bool verbose, size_t workspace_size) {

    Logger logger(verbose);
    _runtime = createInferRuntime(logger);

    bool fp16 = precision.compare("FP16") == 0;
    bool int8 = precision.compare("INT8") == 0;

    // Create builder
    auto builder = createInferBuilder(logger);
    builder->setMaxBatchSize(batch);
    // Allow use of FP16 layers when running in INT8
    builder->setFp16Mode(fp16 || int8);
    builder->setMaxWorkspaceSize(workspace_size);

    // Parse ONNX FCN
    cout << "Building " << precision << " core model..." << endl;
    auto network = builder->createNetwork();
    auto parser = createParser(*network, logger);
    parser->parse(onnx_model, onnx_size);

    auto input = network->getInput(0);
    auto inputDims = input->getDimensions();

    if (int8) {
        builder->setInt8Mode(int8);
        ImageStream stream(batch, inputDims, calibration_images);
        Int8EntropyCalibrator* calib = new Int8EntropyCalibrator(stream, model_name, calibration_table);
        builder->setInt8Calibrator(calib);
    }

    // Add decode plugins
    cout << "Building accelerated plugins..." << endl;
    vector<DecodePlugin> decodePlugins;
    vector<ITensor *> scores, boxes, classes;
    auto nbOutputs = network->getNbOutputs();
    for (int i = 0; i < nbOutputs / 2; i++) {
        auto classOutput = network->getOutput(i);
        auto boxOutput = network->getOutput(nbOutputs / 2 + i);
        auto outputDims = classOutput->getDimensions();

        int scale = inputDims.d[2] / outputDims.d[2];
        auto decodePlugin = DecodePlugin(score_thresh, top_n, anchors[i], scale);
        decodePlugins.push_back(decodePlugin);
        vector<ITensor *> inputs = {classOutput, boxOutput};
        auto layer = network->addPluginV2(inputs.data(), inputs.size(), decodePlugin);
        scores.push_back(layer->getOutput(0));
        boxes.push_back(layer->getOutput(1));
        classes.push_back(layer->getOutput(2));
    }

    // Cleanup outputs
    for (int i = 0; i < nbOutputs; i++) {
        auto output = network->getOutput(0);
        network->unmarkOutput(*output);
    }

    // Concat tensors from each feature map
    vector<ITensor *> concat;
    for (auto tensors : {scores, boxes, classes}) {
        auto layer = network->addConcatenation(tensors.data(), tensors.size());
        concat.push_back(layer->getOutput(0));
    }
    
    // Add NMS plugin
    auto nmsPlugin = NMSPlugin(nms_thresh, detections_per_im);
    auto layer = network->addPluginV2(concat.data(), concat.size(), nmsPlugin);
    vector<string> names = {"scores", "boxes", "classes"};
    for (int i = 0; i < layer->getNbOutputs(); i++) {
        auto output = layer->getOutput(i);
        network->markOutput(*output);
        output->setName(names[i].c_str());
    }

    // Build engine
    cout << "Applying optimizations and building TRT CUDA engine..." << endl;
    _engine = builder->buildCudaEngine(*network);

    // Housekeeping
    parser->destroy();
    network->destroy();
    builder->destroy();

    _prepare();
}

void Engine::save(const string &path) {
    cout << "Writing to " << path << "..." << endl;
    auto serialized = _engine->serialize();
    ofstream file(path, ios::out | ios::binary);
    file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());

    serialized->destroy();    
}

void Engine::infer(vector<void *> &buffers, int batch) {
    _context->enqueue(batch, buffers.data(), _stream, nullptr);
    cudaStreamSynchronize(_stream);
}

vector<int> Engine::getInputSize() {
    auto dims = _engine->getBindingDimensions(0);
    return {dims.d[1], dims.d[2]};
}

int Engine::getMaxBatchSize() {
    return _engine->getMaxBatchSize();
}

int Engine::getMaxDetections() {
    return _engine->getBindingDimensions(1).d[0];
}

int Engine::getStride() {
    return 1;
}

}
