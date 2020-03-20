#include <xerus.h>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <stdlib.h>

#include "../GradientMethods/tangentialOperation.cpp"
#include "../GradientMethods/basic.cpp"
#include "../loading_tensors.cpp"
#include "metropolis.cpp"
#include "contractpsihek.cpp"
#include "trialfunctions.cpp"
#include "probabilityfunctions.cpp"
#include <ctime>

using namespace xerus;
using xerus::misc::operator<<;

Tensor get_test_component(size_t pos, TTTensor phi);
std::vector<size_t> makeRandomSample(size_t p,size_t d);

int main(){
	srand(time(NULL));
	size_t nob = 24;
	size_t num_elec = 8;
	double shift = 25.0;
	size_t position = 15, iterations = 3e8	;
	std::string path_T = "../data/T_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	std::string path_V= "../data/V_H2O_"+std::to_string(2*nob)+"_bench.tensor";
	value_t nuc = -52.4190597253, ref = -76.25663396, res, factor, prob,dk;
	std::vector<size_t> sample = makeRandomSample(num_elec,2*nob),sample1 = { 0, 1,2,3,22,23,30,31 }, sample2,next_sample;
	XERUS_LOG(info,TrialSample(sample,2*nob));


	XERUS_LOG(info, "--- Loading Start Vector ---");
	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);

	Tensor test_component = get_test_component(position,phi);
	//XERUS_LOG(info,"\n"<<test_component);
	XERUS_LOG(info,frob_norm(test_component));


	ContractPsiHek builder(phi,2*nob,num_elec,path_T,path_V,nuc, shift);

  ProjectorProbabilityFunction2 PPF(phi,position,builder);
  Metropolis<ProjectorProbabilityFunction2> markow(PPF, TrialSampleSingle, sample, 2*nob);


	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap;
	XERUS_LOG(info, "P ({ 0, 1,2,3,22,23,30,31 }) = " << PPF.P(sample1));
	XERUS_LOG(info, "- Build MC Chain -");
	XERUS_LOG(info, "Start" << sample);
	for (size_t i = 0; i < iterations/10; ++i)
		next_sample = markow.getNextSample();
	XERUS_LOG(info, "Start" << next_sample);

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow.getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = PPF.P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i % (iterations / 20) == 0)
			XERUS_LOG(info, i);
	}

	size_t count = 0, count1 = 0;
	prob = 0;
	XERUS_LOG(info, "- Calculate component -");
	XERUS_LOG(info, "Number of samples: " << umap.size());

	Tensor result = Tensor(test_component.dimensions),result2 = Tensor(test_component.dimensions),tmp;
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap) {
		builder.reset(pair.first); //setting builder to newest sample!! Important
		res = builder.contract();
		dk = builder.diagionalEntry();
		tmp = PPF.localProduct(pair.first);
		//factor = res/tmp.frob_norm()*((value_t) pair.second.first);
		factor = res/std::pow(tmp.frob_norm()/dk,2)*((value_t) pair.second.first);
		result +=  factor * tmp;
		result2 += res*tmp/dk;
		prob +=std::pow(tmp.frob_norm()/dk,2);
		//Debugging
		count++;
		if (count % 20 == 0)
			XERUS_LOG(info, count);

		if (pair.second.first > iterations/100){
			//XERUS_LOG(info, tmp);
			XERUS_LOG(info, res);
			XERUS_LOG(info, factor / (value_t) iterations);
		}



		if ( pair.second.first > iterations/1000)
			XERUS_LOG(info, "{" << pair.first << "\t: \t" << std::setprecision(3) << pair.second << "|" << res << "}");
	}
	result /= (value_t) iterations;
	result *= prob;



	XERUS_LOG(info,"result of MC integration\n" <<result);
	XERUS_LOG(info,"Norm result  "<<(result).frob_norm());
	XERUS_LOG(info,"Error result "<<(result-test_component).frob_norm());
	XERUS_LOG(info,"result of MC integration2\n" <<result2);
	XERUS_LOG(info,"Norm result2  "<<(result2).frob_norm());
	XERUS_LOG(info,"Error result2 "<<(result2-test_component).frob_norm());
	XERUS_LOG(info,"Sum Prob     "<<prob);


	return 0;
}

Tensor get_test_component(size_t pos, TTTensor phi){
	TTOperator Hs;
	std::vector<Tensor> tang;
	value_t xx,xHx;


	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	read_from_disc("../data/hamiltonian_H2O_48_full_shifted_benchmark.ttoperator",Hs);


	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	XERUS_LOG(info, xHx);
	TangentialOperation top(phi);
	tang = top.localProduct(phi,Hs);
	return tang[pos];
}

std::vector<size_t> makeRandomSample(size_t p,size_t d){
	 std::vector<size_t> sample;
		while(sample.size() < p){
			auto r = rand() % (d);
			auto it = std::find (sample.begin(), sample.end(), r);
			if (it == sample.end()){
				sample.emplace_back(r);
			}
		}
		sort(sample.begin(), sample.end());

		return sample;
}
