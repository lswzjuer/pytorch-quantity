/******************************************************************************
 * Copyright 2018 The Roadstar Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#ifndef MODULES_COMMON_TYPE_MAP_H_
#define MODULES_COMMON_TYPE_MAP_H_

#include <atomic>
#include <unordered_map>
#include <utility>

namespace roadstar {
namespace common {

template <class ValueType>
class TypeMap {
  // Internally, we'll use a hash table to store mapping from type
  // IDs to the values.
  typedef std::unordered_map<int, ValueType> InternalMap;

 public:
  typedef typename InternalMap::iterator iterator;
  typedef typename InternalMap::const_iterator const_iterator;
  typedef typename InternalMap::value_type value_type;

  const_iterator begin() const { return m_map.begin(); }
  const_iterator end() const { return m_map.end(); }
  iterator begin() { return m_map.begin(); }
  iterator end() { return m_map.end(); }

  // Finds the value associated with the type "Key" in the type map.
  template <class Key>
  iterator find() {
    return m_map.find(getTypeId<Key>());
  }

  // Same as above, const version
  template <class Key>
  const_iterator find() const {
    return m_map.find(getTypeId<Key>());
  }

  // Associates a value with the type "Key"
  template <class Key>
  void put(ValueType&& value) {
    m_map[getTypeId<Key>()] = std::forward<ValueType>(value);
  }

 private:
  template <class Key>
  inline static int getTypeId() {
    static const int id = LastTypeId++;
    return id;
  }

  static std::atomic_int LastTypeId;
  InternalMap m_map;
};

template <class ValueType>
std::atomic_int TypeMap<ValueType>::LastTypeId(0);
}  // namespace common
}  // namespace roadstar

#endif  // MODULES_COMMON_TYPE_MAP_H_
