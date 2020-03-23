#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "../classes_old/contractpsihek.cpp"
#include "../classes_old/metropolis.cpp"
#include "../classes_old/probabilityfunctions.cpp"
#include "../classes_old/trialfunctions.cpp"
#include "../GradientMethods/tangentialOperation.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"


using namespace xerus;
using xerus::misc::operator<<;


int main(){
	Index i1,i2;
	size_t nob = 24;
	size_t num_elec = 10;
	double shift = 25.0;
	size_t position = 8, iterations = 100000000;
	std::string path_T = "../data/T_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	value_t nuc = -52.4190597253, ref = -76.25663396, loc_energy, factor;
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 }, next_sample;

	XERUS_LOG(info, "--- Loading Start Vector ---");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	TTOperator Hs;
	value_t xx,xHx;
	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	read_from_disc("../data/hamiltonian_H2O_48_full_benchmark.ttoperator",Hs);

	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	XERUS_LOG(info,xHx);


	ContractPsiHek builder(phi,2*nob,num_elec,path_T,path_V,nuc);

  PsiProbabilityFunction PsiPF(phi);

  Metropolis<PsiProbabilityFunction> markow(PsiPF, TrialSample, sample, 2*nob);

  std::vector<size_t> test_vec = { 0, 1,2,3,22,23,30,31 };
  XERUS_LOG(info,PsiPF.P(test_vec));
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap;
	XERUS_LOG(info, "- Build MC Chain -");
	XERUS_LOG(info, "Start" << sample);

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow.getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = PsiPF.P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i % 10000 == 0)
			XERUS_LOG(info, i);
	}

	size_t count = 0, count1 = 0;
	XERUS_LOG(info, "- Calculate component -");
	XERUS_LOG(info, "Number of samples: " << umap.size());

	value_t result = 0,tmp;
	size_t test_iter = 0;
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap) {
		builder.reset(pair.first); //setting builder to newest sample!! Important
		loc_energy = builder.contract();
		value_t phi_sample = builder.psiEntry();

		result += (loc_energy/phi_sample)*((value_t) pair.second.first);
		count++;
		if (count % 100 == 0)
			XERUS_LOG(info, count);

		//Debuggin
		if(pair.second.first > (value_t)iterations / 1000 ){
			XERUS_LOG(info,"(" << pair.first<< " )");
			XERUS_LOG(info,"(" << loc_energy << ", "  <<phi_sample<< ", " <<(value_t)  pair.second.first / (value_t) iterations << ", " << pair.second.second<< " )");
		}

	}
	XERUS_LOG(info,"result of MC integration\n" <<result);

	result /= (value_t) iterations;



	XERUS_LOG(info,"result of MC integration\n" <<result);
	XERUS_LOG(info,"Norm result  "<<result);
	XERUS_LOG(info,"Error result "<<(result-xHx));



	return 0;
}

