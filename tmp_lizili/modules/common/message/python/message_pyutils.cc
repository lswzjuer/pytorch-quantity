#include <python2.7/Python.h>
#include <functional>
#include "gflags/gflags.h"
#include "modules/common/message/message_service.h"
#include "modules/common/message/tools/message_bag.h"
#include "modules/common/message/tools/proto/message_bag.pb.h"
namespace roadstar {
namespace common {
namespace message {
namespace pyutils {
typedef std::vector<std::string> StringVector;

// NOLINTNEXTLINE
static void Service_initCpp(char *module_name, PyObject *callback_py) {
  Py_INCREF(callback_py);
  auto callback = [callback_py](
                      adapter::AdapterConfig::MessageType message_type,
                      const std::vector<unsigned char> &buffer,
                      bool header_only) {
    PyEval_InitThreads();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyObject *args =
        Py_BuildValue("is#i", message_type, &buffer[0], buffer.size(),
                      static_cast<int>(header_only));
    PyObject *pyinstance = PyObject_CallObject(callback_py, args);
    Py_XDECREF(pyinstance);
    Py_XDECREF(args);
    PyGILState_Release(gstate);
  };
  MessageService::Init(module_name, callback);
  return;
}
static std::unique_ptr<StringVector> ParsePyStringList(PyObject *strlist) {
  StringVector *strvec = new StringVector;
  int list_size = PyList_Size(strlist);
  for (int i = 0; i < list_size; i++) {
    PyObject *stritem = PyList_GetItem(strlist, i);
    std::string str = PyString_AsString(stritem);
    strvec->push_back(str);
  }
  return std::unique_ptr<StringVector>(strvec);
}
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  PyObject_HEAD BagReader *bagreader_; // NOLINT
} message_BagReaderObject; // NOLINT

// NOLINTNEXTLINE
static void BagReader_dealloc(message_BagReaderObject *self) {
  delete self->bagreader_;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// NOLINTNEXTLINE
static PyObject *BagReader_new(PyTypeObject *type, PyObject *args,
                               PyObject *kwds) {
  message_BagReaderObject *self;
  self = reinterpret_cast<message_BagReaderObject *>(type->tp_alloc(type, 0));
  if (self == NULL) return NULL;
  static char *kwlist[] = {"baglist", NULL}; // NOLINT
  PyObject *baglist;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &baglist)) {
    PyErr_SetString(PyExc_TypeError, "Init parameter must be a string list!");
    return NULL;
  }
  Py_XINCREF(baglist);
  std::unique_ptr<StringVector> bagvec = ParsePyStringList(baglist);
  if ((self->bagreader_ = new BagReader(*bagvec)) == NULL) return NULL;
  Py_XDECREF(baglist);
  return reinterpret_cast<PyObject *>(self);
}

// NOLINTNEXTLINE
static PyObject *BagReader_Next(message_BagReaderObject *self, PyObject *args,
                                PyObject *kwds) {
  if (self->bagreader_) {
    PyObject *chunk_py;
    if (!PyArg_ParseTuple(args, "O", &chunk_py)) {
      PyErr_SetString(PyExc_TypeError, "Invalid init parameters!");

      return NULL;
    }
    BagDataChunk bagdata;
    if (!self->bagreader_->Next(&bagdata)) {
      Py_RETURN_FALSE;
    }
    std::string serialized_data;
    bagdata.SerializeToString(&serialized_data);
    PyObject_CallMethod(chunk_py, "ParseFromString", "s#", // NOLINT
                        &(serialized_data)[0], serialized_data.size());
    Py_RETURN_TRUE;
  } else {
    PyErr_SetString(PyExc_StopIteration, "Failed to read next bag.");

    return NULL;
  }
}
// NOLINTNEXTLINE
static PyObject *BagReader_Reset(message_BagReaderObject *self, PyObject *args,
                                 PyObject *kwds) {
  if (self->bagreader_) {
    self->bagreader_->Reset();
    Py_RETURN_NONE;
  }
  return NULL;
}
// NOLINTNEXTLINE
static PyMethodDef BagReader_methods[] = {
    {"Next", (PyCFunction)BagReader_Next, METH_VARARGS, "Read next bag."},
    {"Reset", (PyCFunction)BagReader_Reset, METH_NOARGS,
     "Reset current index."},
    {NULL}};

// NOLINTNEXTLINE
static PyTypeObject message_BagReaderType = {
    PyVarObject_HEAD_INIT(NULL,
                          0) "lib_message_pyutils.BagReader", /* tp_name */
    sizeof(message_BagReaderObject),                          /* tp_basicsize */
    0,                                                        /* tp_itemsize */
    (destructor)BagReader_dealloc,                            /* tp_dealloc */
    0,                                                        /* tp_print */
    0,                                                        /* tp_getattr */
    0,                                                        /* tp_setattr */
    0,                                                        /* tp_compare */
    0,                                                        /* tp_repr */
    0,                                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Message BagReader object.",              /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    BagReader_methods,                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    BagReader_new,                            /* tp_new */
};

typedef struct {
  PyObject_HEAD BagWriter *bagwriter_; // NOLINT
} message_BagWriterObject; // NOLINT
// NOLINTNEXTLINE
static void BagWriter_dealloc(message_BagWriterObject *self) {
  delete self->bagwriter_;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}
// NOLINTNEXTLINE
static PyObject *BagWriter_new(PyTypeObject *type, PyObject *args,
                               PyObject *kwds) {
  message_BagWriterObject *self;
  self = reinterpret_cast<message_BagWriterObject *>(type->tp_alloc(type, 0));
  static char *kwlist[] = {"filename", NULL}; //NOLINT
  char *filename;
  if (self == NULL) return NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filename)) {
    PyErr_SetString(PyExc_TypeError, "Invalid init parameters!");
    return NULL;
  }
  self->bagwriter_ = new BagWriter(filename);

  if (self->bagwriter_ == NULL) {
    return NULL;
  } else {
    return reinterpret_cast<PyObject *>(self);
  }
}
// NOLINTNEXTLINE
static PyObject *BagWriter_FeedData(message_BagWriterObject *self,
                                    PyObject *args, PyObject *kwds) {
  PyObject *chunk_py;
  static char *kwlist[] = {"chunk", NULL}; //NOLINT
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &chunk_py)) {
    PyErr_SetString(PyExc_TypeError, "Invalid init parameters!");
    return NULL;
  }
  Py_XINCREF(chunk_py);
  BagDataChunk chunk;
  chunk.ParseFromString(PyString_AsString(
      PyObject_CallMethod(chunk_py, "SerializeToString", ""))); //NOLINT
  self->bagwriter_->FeedData(chunk);
  Py_XDECREF(chunk_py);
  Py_RETURN_NONE;
}
// NOLINTNEXTLINE
static PyObject *BagWriter_Close(message_BagWriterObject *self, PyObject *args,
                                 PyObject *kwds) {
  self->bagwriter_->Close();
  Py_RETURN_NONE;
}
// NOLINTNEXTLINE
static PyMethodDef BagWriter_methods[] = {
    {"FeedData", (PyCFunction)BagWriter_FeedData, METH_VARARGS,
     "Feed data to the file."},
    {"Close", (PyCFunction)BagWriter_Close, METH_NOARGS, "Close the file."},
    {NULL}};
// NOLINTNEXTLINE
static PyTypeObject message_BagWriterType = {
    PyVarObject_HEAD_INIT(NULL,
                          0) "lib_message_pyutils.BagWriter", /* tp_name */
    sizeof(message_BagWriterObject),                          /* tp_basicsize */
    0,                                                        /* tp_itemsize */
    (destructor)BagWriter_dealloc,                            /* tp_dealloc */
    0,                                                        /* tp_print */
    0,                                                        /* tp_getattr */
    0,                                                        /* tp_setattr */
    0,                                                        /* tp_compare */
    0,                                                        /* tp_repr */
    0,                                                        /* tp_as_number */
    0,                                        /* tp_as_sequence */
    0,                                        /* tp_as_mapping */
    0,                                        /* tp_hash */
    0,                                        /* tp_call */
    0,                                        /* tp_str */
    0,                                        /* tp_getattro */
    0,                                        /* tp_setattro */
    0,                                        /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    "Message BagWriter object.",              /* tp_doc */
    0,                                        /* tp_traverse */
    0,                                        /* tp_clear */
    0,                                        /* tp_richcompare */
    0,                                        /* tp_weaklistoffset */
    0,                                        /* tp_iter */
    0,                                        /* tp_iternext */
    BagWriter_methods,                        /* tp_methods */
    0,                                        /* tp_members */
    0,                                        /* tp_getset */
    0,                                        /* tp_base */
    0,                                        /* tp_dict */
    0,                                        /* tp_descr_get */
    0,                                        /* tp_descr_set */
    0,                                        /* tp_dictoffset */
    0,                                        /* tp_init */
    0,                                        /* tp_alloc */
    BagWriter_new,                            /* tp_new */
};

static PyObject *Init(PyObject *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {"argv", "module_name", "callback", NULL}; //NOLINT
  PyObject *argv;
  char *module_name;
  PyObject *callback;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OsO", kwlist, &argv,
                                   &module_name, &callback)) {
    PyErr_SetString(PyExc_TypeError, "Invalid Parameters!");
    return NULL;
  }
  Py_XINCREF(argv);
  int argc = PyList_Size(argv);
  std::unique_ptr<StringVector> argv_cpp = ParsePyStringList(argv);
  char **argv_c = new char *[argc];
  char **original_argv_c = argv_c;
  for (int i = 0; i < argc; i++) {
    argv_c[i] = &(*argv_cpp)[i][0];
  }
  google::ParseCommandLineFlags(&argc, &argv_c, true);
  roadstar::common::InitLogging(argv_c[0]);
  Py_XDECREF(argv);
  Service_initCpp(module_name, callback);
  delete[] original_argv_c;

  Py_RETURN_NONE;
}

static PyObject *Send(PyObject *self, PyObject *args, PyObject *kwds) {
  int message_type;
  const unsigned char *data;
  int len;
  static char *kwlist[] = {"message_type", "data", NULL}; //NOLINT
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "is#", kwlist, &message_type,
                                   &data, &len)) //NOLINT
    return NULL;
  MessageService::instance()->Send(
      (adapter::AdapterConfig::MessageType)message_type, data, (size_t)len);

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"Init", (PyCFunction)Init, METH_KEYWORDS, "Initialize."},
    {"Send", (PyCFunction)Send, METH_KEYWORDS, "Send."},
    {NULL}};
#ifndef PyMODINIT_FUNC
#define PyMODINIT_FUNC void
#endif
// NOLINTNEXTLINE
PyMODINIT_FUNC initlib_message_pyutils(void) {
  PyObject *mod = PyImport_ImportModule("threading");
  Py_XDECREF(mod);
  PyEval_InitThreads();
  PyObject *m;
  if (PyType_Ready(&message_BagReaderType) < 0) return;
  if (PyType_Ready(&message_BagWriterType) < 0) return;
  m = Py_InitModule3(
      "lib_message_pyutils", module_methods,
      "This modules provides APIs of message_service for python.");
  if (m == NULL) return;
  Py_INCREF(&message_BagReaderType);
  PyModule_AddObject(m, "BagReader",
                     reinterpret_cast<PyObject *>(&message_BagReaderType));
  Py_INCREF(&message_BagWriterType);
  PyModule_AddObject(m, "BagWriter",
                     reinterpret_cast<PyObject *>(&message_BagWriterType));
}

#ifdef __cplusplus
}  // namespace pyutil
#endif
}  // namespace pyutils
}  // namespace message
}  // namespace common
}  // namespace roadstar
