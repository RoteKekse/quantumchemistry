#include <xerus.h>

#include <stdlib.h>



using namespace xerus;
using xerus::misc::operator<<;

class Metropolis{
	public:
		TTTensor psi;
		std::vector <size_t> start_sample;
		size_t d;
		size_t p;
	private:
		std::vector <size_t> current_sample;
		std::vector <size_t> current_sample_inv;
		std::vector <size_t> idx;
		std::vector <size_t> idx_start;

	public:

	Metropolis(TTTensor _psi, std::vector <size_t> _start_sample): psi(_psi), start_sample(_start_sample), p(_start_sample.size()),d(psi.order()), idx(psi.order(),0){
		current_sample = start_sample;
		srand(time(NULL));
		for (size_t i = 0; i < d; ++i)
			if(!std::binary_search (current_sample.begin(), current_sample.end(), i))
				current_sample_inv.emplace_back(i);
			else
				idx[i] = 1;
		idx_start = idx;
		XERUS_LOG(info,idx_start);
	}

	std::vector<size_t> getNextSample(){
		value_t probability_current = psi[idx];
		//XERUS_LOG(info,"---------- Get next Sample ----------");
		//XERUS_LOG(info,"Current Probability: " << probability_current);
		if (current_sample == start_sample){
			auto rand_in = rand() % p;
			auto rand_out = rand() % (d-p);
			idx[current_sample[rand_in]] = 0;
			idx[current_sample_inv[rand_out]] = 1;
			current_sample.clear();
			current_sample_inv.clear();
			for(size_t i = 0; i < d; ++i)
				if (idx[i] == 1)
					current_sample.emplace_back(i);
				else
					current_sample_inv.emplace_back(i);
			if (current_sample != start_sample)
				return current_sample;
		}

		while(true){
			auto rand_in = rand() % p;
			auto rand_out = rand() % (d-p);
			idx[current_sample[rand_in]] = 0;
			idx[current_sample_inv[rand_out]] = 1;
			if (idx == idx_start){
				idx[current_sample[rand_in]] = 1;
				idx[current_sample_inv[rand_out]] = 0;
				continue;
			}


			value_t probability_next = psi[idx];
			//	XERUS_LOG(info,"Next Probability:    " << probability_next);

			value_t r = ((value_t) rand() / (RAND_MAX));
			value_t quotient = probability_next/probability_current;
			if (r < quotient*quotient ){
				current_sample.clear();
				current_sample_inv.clear();
				for(size_t i = 0; i < d; ++i)
					if (idx[i] == 1)
						current_sample.emplace_back(i);
					else
						current_sample_inv.emplace_back(i);
				if (current_sample != start_sample)
					return current_sample;
			}
			else{
				idx[current_sample[rand_in]] = 1;
				idx[current_sample_inv[rand_out]] = 0;
				if (current_sample != start_sample)
					return current_sample;
			}
		}


		//XERUS_LOG(info,"---------- Next Sample: " << current_sample);
		return current_sample;
	}


};
