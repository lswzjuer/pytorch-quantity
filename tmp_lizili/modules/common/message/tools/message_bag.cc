#include "modules/common/message/tools/message_bag.h"

#include <algorithm>
#include <string>
#include <utility>

#include "modules/common/log.h"

namespace roadstar {
namespace common {
namespace message {
namespace {

void ReadProto(std::ifstream* bag, google::protobuf::Message* message) {
  uint64_t len;
  bag->read(reinterpret_cast<char*>(&len), sizeof(len));
  if (bag->fail() || len > (1ull << 32) /* 4G */) {
    AFATAL << "Corrupted bag file";
  }
  len = le64toh(len);
  std::vector<char> buffer(len);
  bag->read(&buffer[0], len);
  if (bag->fail()) {
    AFATAL << "Corrupted bag file";
  }
  if (!message->ParseFromArray(&buffer[0], len)) {
    AFATAL << "Corrupted bag file";
  }
}

void WriteProto(const google::protobuf::Message& message, std::ofstream* bag) {
  uint64_t len = htole64(message.ByteSizeLong());
  bag->write(reinterpret_cast<char*>(&len), sizeof(len));
  message.SerializeToOstream(bag);
}

void WriteString(const std::string& string, std::ofstream* bag) {
  uint64_t len = htole64(string.size());
  bag->write(reinterpret_cast<char*>(&len), sizeof(len));
  bag->write(string.c_str(), len);
}

}  // namespace

BagReader::BagReader(const std::vector<std::string>& bags,
                     const std::unordered_set<int>& ignore_set) {
  for (std::string bag : bags) {
    std::unique_ptr<std::ifstream> bag_file(
        new std::ifstream(bag, std::ios::in | std::ios::binary));
    if (*bag_file) {
      bag_streams_.push_back(std::move(bag_file));
    } else {
      AFATAL << "Not found: " << bag;
    }
  }
  ParseHeader(ignore_set);
}

bool BagReader::Next(BagDataChunk* chunk) {
  if (current_index_ >= global_index_.size()) {
    return false;
  }
  auto index = global_index_[current_index_];
  index.second->seekg(index.first.chunk_offset());
  ReadProto(index.second, chunk);
  current_index_++;
  return true;
}

void BagReader::ResetToTimestamp(uint64_t timestamp) {
  if (global_index_.empty()) {
    return;
  }
  auto target = global_index_.front();
  target.first.mutable_data_header()
        ->set_receive_time_ns(timestamp + GetStartTimeNs());
  auto it = std::lower_bound(global_index_.begin(), global_index_.end(), target,
                [](const std::pair<BagIndex::BagIndexUnit, std::ifstream*>& lhs,
                   const std::pair<BagIndex::BagIndexUnit, std::ifstream*>& rhs) {
                  return lhs.first.data_header().receive_time_ns() <
                         rhs.first.data_header().receive_time_ns();
            });
  Reset(std::distance(global_index_.begin(), it));
}

std::vector<BagIndex> BagReader::GetBagIndex() {
  std::vector<BagIndex> index_proto_vec;
  for (auto& bag_file : bag_streams_) {
    BagHeader header;
    bag_file->seekg(0);
    ReadProto(bag_file.get(), &header);
    bag_file->seekg(header.index_offset());
    BagIndex index_proto;
    ReadProto(bag_file.get(), &index_proto);
    index_proto_vec.push_back(index_proto);
  }
  return index_proto_vec;
}

void BagReader::ParseHeader(const std::unordered_set<int>& ignore_set) {
  for (auto& bag_file : bag_streams_) {
    BagHeader header;
    ReadProto(bag_file.get(), &header);
    start_time_ns_ = std::min(start_time_ns_, header.start_time_ns());
    end_time_ns_ = std::max(end_time_ns_, header.end_time_ns());
    BagIndex index_proto;
    bag_file->seekg(header.index_offset());
    ReadProto(bag_file.get(), &index_proto);
    for (auto unit : index_proto.units()) {
      if (!ignore_set.count(unit.data_header().message_type())) {
        global_index_.push_back(std::make_pair(unit, bag_file.get()));
        type_count_[unit.data_header().message_type()]++;
      }
    }
  }

  std::sort(global_index_.begin(), global_index_.end(),
            [](const std::pair<BagIndex::BagIndexUnit, std::ifstream*>& lhs,
               const std::pair<BagIndex::BagIndexUnit, std::ifstream*>& rhs) {
              return lhs.first.data_header().receive_time_ns() <
                     rhs.first.data_header().receive_time_ns();
            });
}

BagWriter::BagWriter(const std::string& filename) {
  bag_stream_.reset(
      new std::ofstream(filename, std::ios::out | std::ios::binary));
  if (!*bag_stream_) {
    AFATAL << "open " << filename << " fial";
  }
  // Header placeholder
  std::vector<char> empty(1024);
  bag_stream_->write(&empty[0], 1024);
}

void BagWriter::Close() {
  if (!bag_stream_) {
    return;
  }
  uint64_t index_offset = bag_stream_->tellp();
  WriteProto(bag_index_, bag_stream_.get());
  BagHeader bag_header;
  bag_header.set_index_offset(index_offset);
  bag_header.set_start_time_ns(start_time_ns_);
  bag_header.set_end_time_ns(end_time_ns_);
  bag_stream_->seekp(0, std::ios_base::beg);
  WriteProto(bag_header, bag_stream_.get());
  bag_stream_->close();
  bag_stream_.reset();
}

void BagWriter::FeedData(const BagDataChunk& chunk) {
  start_time_ns_ =
      std::min(chunk.data_header().receive_time_ns(), start_time_ns_);
  end_time_ns_ = std::max(chunk.data_header().receive_time_ns(), end_time_ns_);
  
  std::string string;
  chunk.SerializeToString(&string);
  uint64_t offset = string.substr(chunk.data_header().ByteSizeLong())
                          .find(chunk.message_data());
  auto* unit = bag_index_.add_units();
  unit->mutable_data_header()->CopyFrom(chunk.data_header());
  unit->set_chunk_offset(bag_stream_->tellp());
  unit->set_message_data_length(chunk.message_data().size());
  unit->set_message_data_offset(unit->chunk_offset() + sizeof(uint64_t) +
                                chunk.data_header().ByteSizeLong() + offset);
  WriteString(string, bag_stream_.get());
}

}  // namespace message
}  // namespace common
}  // namespace roadstar
