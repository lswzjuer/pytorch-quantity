#include "modules/common/message/tools/message_bag.h"

#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "modules/common/log.h"
#include "modules/common/message/tools/proto/message_bag.pb.h"


namespace roadstar {
namespace common {
namespace message {
namespace {
void WriteBag(const std::string& path,
                const std::vector<BagDataChunk>& chunks) {
  BagWriter writer(path);
  for (auto chunk : chunks) {
    writer.FeedData(chunk);
  }
  writer.Close();
}

}  // namespace

class MessageBagTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    template_ = "bag.XXXXXX";
  }
  virtual void TearDown() {
    for (auto filename : fileneams_) {
      unlink(filename.c_str());
    }
  }
  std::string GetFilename() {
    char filename[100];
    snprintf(filename, sizeof(filename), "%s", template_.c_str());
    mktemp(filename); //NOLINT
    fileneams_.push_back(std::string(filename));
    return fileneams_.back();
  }

  std::string template_;
  std::vector<std::string> fileneams_;
};

TEST_F(MessageBagTest, RaedWriteBag) {
  std::string filename = GetFilename();
  BagDataChunk chunk;

  std::vector<BagDataChunk> chunks;
  std::vector<char> buffers{'a', 'b', 'c', 'd', 'e'};
  for (size_t i = 0; i != buffers.size(); i++) {
    chunk.mutable_data_header()->set_message_type(i + 1);
    chunk.mutable_data_header()->set_receive_time_ns(i * 100);
    chunk.set_message_data(&buffers[i], 1);
    chunks.push_back(chunk);
  }
  WriteBag(filename, chunks);
  BagReader reader(std::vector<std::string>{filename});
  EXPECT_EQ(reader.GetStartTimeNs(), 0);
  EXPECT_EQ(reader.GetEndTimeNs(), 400);
  for (size_t i = 0; i != buffers.size(); i++) {
    EXPECT_TRUE(reader.Next(&chunk));
    EXPECT_EQ(chunk.data_header().message_type(), i + 1);
    EXPECT_EQ(chunk.data_header().receive_time_ns(), i * 100);
    EXPECT_EQ(chunk.message_data(), std::string{buffers[i]});
  }
  EXPECT_FALSE(reader.Next(&chunk));

  reader.ResetToTimestamp(100);
  EXPECT_TRUE(reader.Next(&chunk));
  EXPECT_EQ(chunk.data_header().receive_time_ns(), 100);
  reader.Reset(2);
  EXPECT_TRUE(reader.Next(&chunk));
  EXPECT_EQ(chunk.data_header().receive_time_ns(), 200);

  reader.ResetToTimestamp(1000);
  EXPECT_FALSE(reader.Next(&chunk));

  auto index_vec = reader.GetBagIndex();
  EXPECT_EQ(index_vec.size(), 1);
  std::ifstream bag_stream(filename, std::ios::in | std::ios::binary);
  for (size_t i = 0; i != index_vec[0].units().size(); i++) {
    auto& unit = index_vec[0].units()[i];
    std::string string(unit.message_data_length(), '\0');
    bag_stream.seekg(unit.message_data_offset());
    bag_stream.read(&string[0], unit.message_data_length());
    EXPECT_EQ(string, chunks[i].message_data());
  }
}

TEST_F(MessageBagTest, RaedWriteBigChunk) {
  std::string filename = GetFilename();
  BagDataChunk chunk;

  std::vector<BagDataChunk> chunks;
  std::vector<char> buffer;
  for (size_t i = 0; i < 1<<22; i++) {
    buffer.push_back('a');
  }

  chunk.mutable_data_header()->set_message_type(1);
  chunk.mutable_data_header()->set_receive_time_ns(100);
  chunk.set_message_data(&buffer[0], buffer.size());
  chunks.push_back(chunk);
  WriteBag(filename, chunks);

  BagReader reader(std::vector<std::string>{filename});
  buffer.push_back('\0');
  EXPECT_TRUE(reader.Next(&chunk));
  EXPECT_EQ(chunk.message_data(), &buffer[0]);
}

TEST_F(MessageBagTest, ReadMultiBag) {
  std::vector<std::string> filenames;
  filenames.push_back(GetFilename());
  filenames.push_back(GetFilename());
  BagDataChunk chunk;

  std::vector<BagDataChunk> chunks;
  std::vector<char> buffers{'a', 'b', 'c', 'd', 'e'};
  for (size_t i = 0; i != filenames.size(); i++) {
    for (size_t j = 0; j != buffers.size(); j++) {
      chunk.mutable_data_header()->set_message_type(j + 1);
      chunk.mutable_data_header()->set_receive_time_ns(i * 50 + j * 100);
      chunk.set_message_data(&buffers[j], 1);
      chunks.push_back(chunk);
    }
    WriteBag(filenames[i], chunks);
    chunks.clear();
  }
  BagReader reader(filenames);
  EXPECT_EQ(reader.GetStartTimeNs(), 0);
  EXPECT_EQ(reader.GetEndTimeNs(), 450);
  for (size_t j = 0; j != buffers.size() * 2; j++) {
    EXPECT_TRUE(reader.Next(&chunk));
    EXPECT_EQ(chunk.data_header().message_type(), (j / 2) + 1);
    EXPECT_EQ(chunk.data_header().receive_time_ns(), j * 50);
    EXPECT_EQ(chunk.message_data(), std::string{buffers[j / 2]});
  }
  EXPECT_FALSE(reader.Next(&chunk));
}

}  // namespace message
}  // namespace common
}  // namespace roadstar


