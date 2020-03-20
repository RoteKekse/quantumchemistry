#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include "../GradientMethods/tangentialOperation.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"
#include "metropolis.cpp"
#include "contractpsihek.cpp"
#include "trialfunctions.cpp"
#include "probabilityfunctions.cpp"


using namespace xerus;
using xerus::misc::operator<<;

Tensor get_test_component(size_t pos, TTTensor phi);

int main(){
	size_t nob = 24;
	size_t num_elec = 10;
	double shift = 25.0;
	size_t position = 47, iterations = 1e8;
	std::string path_T = "../data/T_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	value_t nuc = -52.4190597253, ref = -76.25663396, res, factor, prob;
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 }, next_sample;

	XERUS_LOG(info, "--- Loading Start Vector ---");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);

	Tensor test_component = get_test_component(position,phi);
	XERUS_LOG(info,"\n"<<test_component);
	XERUS_LOG(info,frob_norm(test_component));


	ContractPsiHek builder(phi,2*nob,num_elec,path_T,path_V,nuc);

  PsiProbabilityFunction PsiPF(phi);
  ProjectorProbabilityFunction PPF(phi,position);
  Metropolis<PsiProbabilityFunction> markow(PsiPF, TrialSample, sample, 2*nob);


	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap;
	XERUS_LOG(info, "- Build MC Chain -");
	XERUS_LOG(info, "Start" << sample);
	for (size_t i = 0; i < iterations/1000; ++i)
		next_sample = markow.getNextSample();
	XERUS_LOG(info, "Start" << next_sample);

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow.getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = PsiPF.P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i % 1000000 == 0)
			XERUS_LOG(info, i);
	}

	size_t count = 0, count1 = 0;
	XERUS_LOG(info, "- Calculate component -");
	XERUS_LOG(info, "Number of samples: " << umap.size());

	Tensor result = Tensor(test_component.dimensions),tmp;
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap) {
		builder.reset(pair.first); //setting builder to newest sample!! Important
		res = builder.contract(); // NOTE: no shift!!
		tmp = PPF.localProduct(pair.first);
		factor = res/std::pow(tmp.frob_norm(),2)*((value_t) pair.second.first);
		//factor = res/tmp.frob_norm()*((value_t) pair.second.first);
		result +=  factor * tmp;

		//Debugging
		count++;
		if (count % 100 == 0)
			XERUS_LOG(info, count);

		if (pair.second.first > iterations/100){
			XERUS_LOG(info, tmp);
			XERUS_LOG(info, res);
			XERUS_LOG(info, factor);
		}



		if ( pair.second.first > iterations/1000)
			XERUS_LOG(info, "{" << pair.first << "\t: \t" << std::setprecision(3) << pair.second << "|" << res << "}");
	}
	result /= (value_t) iterations;



	XERUS_LOG(info,"result of MC integration\n" <<result);
	XERUS_LOG(info,"Norm result  "<<(result).frob_norm());
	XERUS_LOG(info,"Error result "<<(result-test_component).frob_norm());


	return 0;
}

Tensor get_test_component(size_t pos, TTTensor phi){
	TTOperator Hs;
	std::vector<Tensor> tang;
	value_t xx,xHx;


	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	read_from_disc("../data/hamiltonian_H2O_48_full_benchmark.ttoperator",Hs);


	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	XERUS_LOG(info, xHx);
	TangentialOperation top(phi);
	tang = top.localProduct(phi,Hs);
	return tang[pos];
}
