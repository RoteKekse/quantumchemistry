#include <xerus.h>
#include <boost/functional/hash.hpp>
#include <stdlib.h>
#include <unordered_map>

#pragma once

template <typename Container>
struct container_hash {
    std::size_t operator()(Container const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};
