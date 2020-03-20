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
#include "unitvectorprojection.cpp"

Tensor get_test_component(size_t pos, TTTensor phi,value_t& ev);
Tensor get_test_component2(size_t pos, TTTensor phi);
std::vector<size_t> makeRandomSample(size_t p,size_t d);
TTOperator particleNumberOperator(size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
std::vector<size_t> makeIndex(std::vector<size_t> sample,size_t d);

template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow,std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations);


int main(){
	srand(time(NULL));
	size_t d = 48, p = 8, position = 20, iterations = 1e6;
	bool proj = true;
	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";
	value_t nuc = -52.4190597253, ref = -76.25663396, shift = 25.0, res, factor, prob=0,dk,psi_ek,ev_exact;
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 },sample1 = { 0, 1,2,3,22,23,30,31 }, sample2;
	//sample = { 2, 3, 5, 6, 22, 23, 30, 31 };
	//sample = makeRandomSample(p,d);
	XERUS_LOG(info, "--- Loading Start Vector ---");
	TTTensor phi,TTtest;
	TTOperator Projector = TTOperator::identity(std::vector<size_t>(2*d,2)), id = TTOperator::identity(std::vector<size_t>(2*d,2));
	Index i1,i2,j1,j2,k1,k2;

	phi = TTTensor::dirac(std::vector<size_t>(d,2),makeIndex(sample,d));
	XERUS_LOG(info, sample << " " << phi[makeIndex(sample,d)]);
	for (size_t i = 0; i < 1; ++i){
		//sample2 = TrialSample(sample,d);
		sample2 =  {0,2,3,20,22,23,30,31 };
		phi +=  TTTensor::dirac(std::vector<size_t>(d,2),makeIndex(sample2,d));
		XERUS_LOG(info, sample2 << " " << phi[makeIndex(sample2,d)]);
	}
	phi.move_core(0);
	phi/= phi.frob_norm();

//	read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);
	//Projection of phi onto eigenspace
	TTtest = phi;
	for (size_t k = 0; k <= d; ++k){
		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t)p - (value_t) k;
			phi /=  f;
			phi.round(1e-12);
		}
	}
	phi.round(1e-6);
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	XERUS_LOG(info, "Phi rounding error " << (TTtest-phi).frob_norm());
	XERUS_LOG(info,phi.ranks());




	Tensor test_component = get_test_component(position,phi,ev_exact);
	XERUS_LOG(info, "Test norm        " << frob_norm(test_component));

	ContractPsiHek builder(phi,d,p,path_T,path_V,nuc, shift);
  PsiProbabilityFunction PsiPF(phi);
  ProjectorProbabilityFunction PPF(phi,position, proj);
  Metropolis<PsiProbabilityFunction> markow1(&PsiPF, TrialSample, sample1, d);
  Metropolis<ProjectorProbabilityFunction> markow2(&PPF, TrialSample, sample, d);
  unitVectorProjection uvP(phi,position);

  sample2 = {0,2,3,20,22,23,30,31 }; // 8
  XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
  sample2 ={0,1,2,3,20,22,23,30,31 }; // 9
	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
//	sample2 = {0,1,2,3,20,22,23,30,31 }; //9
//	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
//	sample2 = {0,2,3,20,22,23,30,31 } ; //9
//	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));

	sample2 = {0,1,2,3,22,23,30,31 }; //8
	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
	sample2 = {0,2,3,22,23,30,31 } ; //7
	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
//	sample2 = {0,1,2,3,22,23,30,31 }; //9
//	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));
//	sample2 = {0,2,3,22,23,30,31 } ; //7
//	XERUS_LOG(info, sample2 << " " << PPF.P(sample2) << "\n" << uvP.localProduct(sample2,proj));

	return 0;

	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap1;
	std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> umap2;
	XERUS_LOG(info, "P ({ 0, 1,2,3,22,23,30,31 }) = " << PsiPF.P(sample1));
	XERUS_LOG(info, "Run Metropolis, start sample: " << sample1);

	runMetropolis<PsiProbabilityFunction>(&markow1,umap1,iterations);
  runMetropolis<ProjectorProbabilityFunction>(&markow2,umap2,iterations);





	value_t ev = 0;
	//Eigenvalue
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap1) {
		builder.reset(pair.first);
		res = builder.contract();
		psi_ek = PsiPF.values[pair.first];
		factor = res* (value_t) pair.second.first/psi_ek;
		ev += factor;
		XERUS_LOG(info, "{" << pair.first << ": " << std::setprecision(3) << pair.second  << " " << res << "}");
	}
	ev /= (value_t) iterations;





	XERUS_LOG(info,"Caluclate expectation of gradient");
	size_t count = 0;
	value_t sum = 0;
	Tensor result(test_component.dimensions);
	for (std::pair<std::vector<size_t>,std::pair<size_t,value_t>> const& pair: umap2) {
		builder.reset(pair.first); //setting builder to newest sample!! Important
		res = builder.contract();
		psi_ek = PPF.values[pair.first];
		sum +=psi_ek*psi_ek;
		auto loc_grad = uvP.localProduct(pair.first,proj);

		auto idx = PsiPF.makeIndex(pair.first);
		factor = (res - ev*phi[idx])* (value_t) pair.second.first/(psi_ek*psi_ek);
		result += factor * loc_grad;
		prob +=pair.second.second; //TODO check this

		//Debugging
		count++;
		if (count % 20 == 0)
			XERUS_LOG(info, count);
		if ( pair.second.first > iterations/10000)
			XERUS_LOG(info, "{" << pair.first << ": " << std::setprecision(3) << pair.second  << " " << res << "}");
	}
	result /= (value_t) iterations;

	XERUS_LOG(info, "Number of probs: " <<  PsiPF.values.size());
	XERUS_LOG(info, "Number of samples umap1: " << umap1.size());
	XERUS_LOG(info, "Number of samples umap2: " << umap2.size());

	XERUS_LOG(info, "Exact ev    " << ev_exact);
	XERUS_LOG(info,"Ev           "<<ev);
	XERUS_LOG(info,"Ev error     "<<std::abs(ev - ev_exact));

	XERUS_LOG(info,"Sum psi_ek   "<<sum);
	XERUS_LOG(info,"prob         "<< prob);
	XERUS_LOG(info,"test component: "<< test_component.frob_norm() << "\n" << test_component);
	XERUS_LOG(info,"result*prob:    "<< (result*prob).frob_norm() << "\n" << result*prob);
	XERUS_LOG(info,"test component - result*prob:    "<< (test_component-result*prob).frob_norm() << "\n" << test_component-result*prob);



	return 0;
}


TTOperator particleNumberOperator(size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n,{0,0,0,1});
	op.set_component(0,tmp);
	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}

TTOperator particleNumberOperator(size_t k, size_t d){
	TTOperator op(std::vector<size_t>(2*d,2));
	Tensor id = Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto n = id;
	n[{0,0,0,0}] = 0;

	value_t kk = (value_t) k;
	auto kkk = kk/(value_t) d * id;
	Tensor tmp({1,2,2,2});
	tmp.offset_add(id,{0,0,0,0});
	tmp.offset_add(n-kkk,{0,0,0,1});
	op.set_component(0,tmp);

	for (size_t i = 1; i < d-1; ++i){
		tmp = Tensor({2,2,2,2});
		tmp.offset_add(id,{0,0,0,0});
		tmp.offset_add(id,{1,0,0,1});
		tmp.offset_add(n-kkk,{0,0,0,1});
		op.set_component(i,tmp);
	}
  tmp = Tensor({2,2,2,1});
	tmp.offset_add(n-kkk,{0,0,0,0});
	tmp.offset_add(id,{1,0,0,0});
	op.set_component(d-1,tmp);


	return op;
}



template<class ProbabilityFunction>
void runMetropolis(Metropolis<ProbabilityFunction>* markow, std::unordered_map<std::vector<size_t>,std::pair<size_t,value_t>,container_hash<std::vector<size_t>>> &umap, size_t iterations){
	std::vector<size_t> next_sample;
	XERUS_LOG(info, "- Build MC Chain -");
	for (size_t i = 0; i < iterations/10; ++i)
		next_sample = markow->getNextSample();
	XERUS_LOG(info, "Start" << next_sample);

	for (size_t i = 0; i < iterations; ++i){
		next_sample = markow->getNextSample();
		auto itr = umap.find(next_sample);
		if (itr == umap.end()){
			umap[next_sample].first = 1;
			umap[next_sample].second = markow->P->P(next_sample);
		} else
			umap[next_sample].first += 1;
		if (i % (iterations / 10) == 0)
			XERUS_LOG(info, i);
	}
}




Tensor get_test_component2(size_t pos, TTTensor phi){
	std::vector<Tensor> tang;
	TTOperator id = TTOperator::identity(std::vector<size_t>(2*phi.order(),2));
	value_t xx;

	xx = phi.frob_norm();
	phi /= xx; //normalize
	TangentialOperation top(phi);

	tang = top.localProduct(phi,id);
	return tang[pos];
}


Tensor get_test_component(size_t pos, TTTensor phi, value_t& ev){
	TTOperator Hs;
	std::vector<Tensor> tang;
	value_t xx,xHx;


	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	read_from_disc("../data/hamiltonian_H2O_48_full_shifted_benchmark.ttoperator",Hs);


	xx = phi.frob_norm();
	phi /= xx; //normalize
	xHx = contract_TT(Hs,phi,phi);
	ev = xHx;
	TangentialOperation top(phi);
	TTOperator id = TTOperator::identity(std::vector<size_t>(2*phi.order(),2));
	tang = top.localProduct(Hs,id,xHx,false);
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

std::vector<size_t> makeIndex(std::vector<size_t> sample,size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	return index;
}
