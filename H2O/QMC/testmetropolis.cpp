#include <xerus.h>
#include <chrono>

#include "../classes/QMC/metropolis.cpp"
#include "../classes/QMC/contractpsihek.cpp"
#include "../classes/QMC/trialfunctions.cpp"
#include "../classes/QMC/probabilityfunctions.cpp"
#include "../classes/QMC/unitvectorprojection.cpp"

template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations);


int main(){
	size_t d = 48, iterations = 1e7;

	std::vector<size_t> start_sample = {0,1,2,3,22,23,30,31};
	xerus::TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	PsiProbabilityFunction PsiPF(phi);
	Metropolis<PsiProbabilityFunction> markow1(&PsiPF, TrialSample2, start_sample, d);
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;

	auto start = std::chrono::steady_clock::now();
	runMetropolis<PsiProbabilityFunction>(&markow1,samples,iterations);
	auto end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for linear evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");


	return 0;
}


template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations){
	std::vector<size_t> next_sample;
	for (size_t i = 0; i <  (size_t) (iterations/10); ++i)
		next_sample = markow->getNextSample();

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow->getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = markow->P->P(next_sample);
		} else
			umap[next_sample].first += 1;

	}
}
