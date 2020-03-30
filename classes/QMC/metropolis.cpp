#include <xerus.h>
#include "probabilityfunctions.cpp"
#include <stdlib.h>
#include <unordered_map>
#include "../../classes/containerhash.cpp"

using namespace xerus;
using xerus::misc::operator<<;



template<class ProbabilityFunction>
class Metropolis{
	public:
		size_t d,p; // d problem size, p number of  particles
		ProbabilityFunction* P; // Give a sample get a probability, needed for acceptance
	private:
		std::function<std::vector<size_t>(std::vector<size_t>,size_t)> T; // Trial function to move form one sample to another, should keep p constant
		std::vector <size_t> current_sample;
		value_t probability_current;
	public:

	Metropolis(ProbabilityFunction* _P, std::function<std::vector<size_t>(std::vector<size_t>,size_t)> _T,
			std::vector <size_t> _start_sample, size_t _d):
		P(_P), T(_T), p(_start_sample.size()), d(_d), current_sample(_start_sample)
	{
		srand(time(NULL));
		probability_current = P->P(current_sample);
	}

	Metropolis( const Metropolis&  _other ) = default;

	std::vector<size_t> getNextSample(){

		std::vector<size_t> next_sample(T(current_sample,d));
		value_t probability_next = P->P(next_sample);
		if (probability_next > 1e-8)
			XERUS_LOG(info,next_sample << "  \t " << probability_next);

		value_t random_number = ((value_t) rand() / (RAND_MAX));
		value_t acceptance_rate = probability_next/probability_current;

//		XERUS_LOG(info, "Current P       "<<probability_current);
//		XERUS_LOG(info, "Next    P       "<<probability_next);
//		XERUS_LOG(info, "Random Number   "<<random_number);
//		XERUS_LOG(info, "Acceptance rate "<<acceptance_rate);

		if (random_number < acceptance_rate ){
			current_sample = std::move(next_sample);
			probability_current = probability_next;
		}
		return current_sample;
	}
};
