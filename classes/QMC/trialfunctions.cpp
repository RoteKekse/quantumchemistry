#include <xerus.h>
#include <stdlib.h>

#pragma once

void addElementToVector(std::vector<size_t>& v, size_t elem, size_t max_size);


std::vector<size_t> TrialSample(std::vector<size_t> sample, size_t dim){
	auto rand_in = rand() % sample.size();
	while(true){
		size_t rand_out = rand() % dim;

		auto found = std::binary_search (sample.begin(), sample.end(), rand_out);
			if (!found){
				sample.erase (sample.begin()+rand_in);
				addElementToVector(sample, rand_out, dim);
				break;
			}
	}

	return sample;
}

std::vector<size_t> TrialSampleSingle(std::vector<size_t> sample, size_t dim){
	auto rand_up = rand() % 3;
	if (rand_up == 1){
		while(true){
			auto rand_new = rand() % dim;
	    auto it = std::find (sample.begin(), sample.end(), rand_new);
			if (it == sample.end()){
				sample.emplace_back(rand_new);
				break;
			}
		}
	} else if (rand_up == 2){
		auto rand_new = rand() % sample.size();
		sample.erase(sample.begin() + rand_new);
	} else {
		sample = TrialSample(sample,dim);
	}

	sort(sample.begin(), sample.end());
	return sample;
}


//TODO check if this is working correctly
std::vector<size_t> TrialSample2(std::vector<size_t> sample, size_t dim){
	auto sample_new = TrialSample(sample, dim);
	if  (rand() % 2 == 1) return sample_new; //!!!
	while(true){
		sample_new = TrialSample(sample_new, dim);
		if ( sample_new != sample)
			return sample_new;
	}
}


void addElementToVector(std::vector<size_t>& v, size_t elem, size_t max_size) {
  if( max_size > v.size() ) { // if empty room in v
    // add room at the end:
    v.push_back(elem);
  };
  // find the position for the element
  auto pos = std::upper_bound(v.begin(), v.end()-1, elem);
  // and move the array around it:

  std::move_backward(pos, v.end()-1, v.end());

  // and set the new element:
  *pos = elem;
};
