#include <xerus.h>
#include "../loading_tensors.cpp"
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include "../classes_old/metropolis.cpp"
#include "../classes_old/probabilityfunctions.cpp"
#include "../classes_old/trialfunctions.cpp"

using namespace xerus;
using xerus::misc::operator<<;




int main(){
	srand(time(NULL));

	size_t dim = 10;
	size_t iterations = 10;
	std::vector<size_t> test = {2,4,6,8};


	XERUS_LOG(info, "----- Test with dummy probability function -----");

	DummyProbabilityFunction P_cont;
	Metropolis<DummyProbabilityFunction> markow(P_cont, TrialSample, test, dim);

	for (size_t i = 0; i < iterations; ++i)
		XERUS_LOG(info,markow.getNextSample());

	XERUS_LOG(info, "----- Test with projected components -----");
	size_t pos = 8;
	iterations = 1000;
	dim = 50;
	test = { 0, 1, 2, 7, 8, 22, 23, 31, 32, 48 };
	std::vector<size_t> next_sample;

	XERUS_LOG(info, "- Loading Test Vector -");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	ProjectorProbabilityFunction PPF(phi,pos);
	Metropolis<ProjectorProbabilityFunction> markow2(PPF, TrialSample, test, dim);



	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap;

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow2.getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = PPF.P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i % 10000 == 0)
			XERUS_LOG(info, i);
	}

	XERUS_LOG(info, "- Statistics -");
	size_t count = 0, count1 = 0;
	for (auto const& pair: umap) {
		if (pair.second.first > 1000){
			XERUS_LOG(info, "{" << pair.first << "\t: \t" << pair.second << "}");
			count1++;
		}
		count++;
	}

	XERUS_LOG(info,"Number of entries                   " << count);
	XERUS_LOG(info,"Number of entries greater than 1000 " << count1);
	return 0;
}
