#include <xerus.h>
#include <chrono>

#include "../../classes/QMC/metropolis.cpp"
#include "../../classes/QMC/contractpsihek.cpp"
#include "../../classes/QMC/trialfunctions.cpp"
#include "../../classes/QMC/probabilityfunctions.cpp"
#include "../../classes/QMC/unitvectorprojection.cpp"

template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations);


int main(){
	size_t d = 48, iterations = 1e7;

	std::vector<size_t> start_sample = {0,1,2,3,22,23,30,31};
	xerus::TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	XERUS_LOG(info,"Test TrialSampleSym2");
	XERUS_LOG(info,TrialSampleSym2(start_sample,d));
	XERUS_LOG(info,TrialSampleSym2(start_sample,d));
	XERUS_LOG(info,TrialSampleSym2(start_sample,d));
	XERUS_LOG(info,TrialSampleSym2(start_sample,d));

	XERUS_LOG(info, "Start metropolis tree");
	PsiProbabilityFunction2 PsiPF2(phi);
	Metropolis<PsiProbabilityFunction2> markow2(&PsiPF2, TrialSampleSym2, start_sample, d);
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples2;
	auto start = std::chrono::steady_clock::now();
	runMetropolis<PsiProbabilityFunction2>(&markow2,samples2,iterations);
	auto end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for tree evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");


	XERUS_LOG(info, "Start metropolis linear");
	PsiProbabilityFunction PsiPF(phi);
	Metropolis<PsiProbabilityFunction> markow1(&PsiPF, TrialSampleSym2, start_sample, d);
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;
	start = std::chrono::steady_clock::now();
	runMetropolis<PsiProbabilityFunction>(&markow1,samples,iterations);
	end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds for linear evaluation: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< " msec");





	return 0;
}


template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations){
	std::vector<size_t> next_sample;
//	for (size_t i = 0; i <  (size_t) (iterations/10); ++i){
//		next_sample = markow->getNextSample();
//		if (i%(iterations/100) == 0)
//				XERUS_LOG(info,"Step " << i);
//	}

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow->getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			XERUS_LOG(info,"The current sample is " << next_sample);
			umap[next_sample].first = 1;
			umap[next_sample].second = markow->P->P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i%(iterations/10) == 0){
			XERUS_LOG(info,"Step " << i);
			XERUS_LOG(info,"Size umap " << umap.size());
		}
	}
}
