#include <xerus.h>
#include <chrono>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/trialfunctions.cpp"
#include "../../classes/QMC/tangential_parallel.cpp"



int main(){
	size_t d = 48,p = 8,iterations = 1e5,iterations2 = 100*iterations, rank = 10;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=0.6,ev_app_tmp;
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	value_t nuc = -52.4190597253;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	value_t alpha = 0.1,beta;

	xerus::TTTensor phi,res,res_last,start;
	phi = makeUnitVector(hf_sample,d);
	read_from_disc("../data/hf_gradient_48.tttensor",start);
	XERUS_LOG(info,"Round start vector to " << eps << " keeping sing values bigger than " << eps/std::sqrt(d-1));
	start/= start.frob_norm();
	for (value_t ee = 0.05; ee <= eps ; ee+=0.05){
		start.round(ee);
		start/= start.frob_norm();
	}
	auto P = particleNumberOperator(d);
	auto Pup = particleNumberOperatorUp(d);
	auto Pdown = particleNumberOperatorDown(d);

	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));

	phi -= alpha*start;
	phi/=phi.frob_norm();
	phi.move_core(0);
	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));
	XERUS_LOG(info,phi.ranks());

	Tangential tang(d,p,iterations,path_T,path_V,shift,hf_sample,phi);
	ev = tang.get_eigenvalue();
	XERUS_LOG(info,"Approximated Eigenvalue: " << ev - shift + nuc);

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	XERUS_LOG(info,"Exact Eigenvalue: " << contract_TT(Hs,phi,phi) - shift + nuc);



	int eHx_time =0, next_time=0;
	std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> eHxValues;
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> samples;
	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift,hf_sample);
	value_t probability_next,probability_current,random_number,acceptance_rate, psi_ek,factor;
	std::vector<size_t> sample = hf_sample,next;
	builder.reset(sample);
	builder.preparePsiEval();
	value_t tmp = builder.contract_tree();
	eHxValues[sample] = tmp;
	probability_current = std::pow(builder.umap_psi_tree[makeIndex(sample,d)],2);
	for (size_t i = 0; i < iterations2; ++i){
		next = TrialSampleSym2(sample,d);
		auto itr = builder.umap_psi_tree.find(makeIndex(sample,d));
		if (itr == builder.umap_psi_tree.end())
			XERUS_LOG(info,"Sample " << next << " not found");

		probability_next = std::pow(builder.umap_psi_tree[makeIndex(next,d)],2);
		random_number = ((value_t) rand() / (RAND_MAX));
		acceptance_rate = probability_next/probability_current;

		if (random_number < acceptance_rate ){
			sample = std::move(next);
			probability_current = probability_next;

			auto itr_eHx = eHxValues.find(sample);
			if (itr_eHx == eHxValues.end()){
				auto eHx_start = std::chrono::steady_clock::now();
				builder.reset(sample);
				builder.preparePsiEval();
				tmp = builder.contract_tree();
				eHxValues[sample] = tmp;
				auto eHx_end = std::chrono::steady_clock::now();
				eHx_time += std::chrono::duration_cast<std::chrono::microseconds>(eHx_end - eHx_start).count();
			}
		}

		auto itr2 = samples.find(sample);
		if (itr2 == samples.end()){
			samples[sample].first = 1;
			samples[sample].second = std::pow(builder.umap_psi_tree[makeIndex(sample,d)],2);
		} else
			samples[sample].first += 1;
	}


	value_t ev2 = 0;
	XERUS_LOG(info, "Number of samples for Eigenvalue " << samples.size());
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: samples) {
		psi_ek = builder.umap_psi_tree[makeIndex(pair.first,d)];
		factor = eHxValues[pair.first]*psi_ek;
		ev2 += factor;

	}
	XERUS_LOG(info,"Approximated Eigenvalue2: " << ev2 - shift + nuc);

	XERUS_LOG(info, "Elapsed time in seconds for eHx: "
			<< ((value_t) eHx_time) / 1e6
			<< " musec");

	return 0;
}





