#include <xerus.h>
#include <chrono>

#include "classes/contractpsihek.cpp"
#include "classes/trialfunctions.cpp"

#include "../loading_tensors.cpp"

using namespace xerus;
using xerus::misc::operator<<;

TTOperator particleNumberOperator(size_t k, size_t d);
void project(TTTensor &phi, size_t p, size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);

int main(){
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	size_t shift = 25.0,d = 48, p = 8;

	XERUS_LOG(info, (size_t) 5 / (size_t) 2 );

	size_t test_number = 1;
	size_t test_number2 = 0;
	std::vector<size_t> sample = {0,1,2,3,22,23,30,31};

	XERUS_LOG(info, "--- Loading shifted and preconditioned Hamiltonian ---");
	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	xerus::TTTensor phi,res,res_last;
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	//read_from_disc("../data/residual_app_" + std::to_string(2*nob)  +"_benchmark_diag_one_step.tttensor",phi);
	phi /= phi.frob_norm(); //normalize
	project(phi,p,d);


	TensorNetwork phitmp = phi;
	phitmp.fix_mode(46,0);
	phitmp.fix_mode(45,1);
	phitmp.fix_mode(44,0);
	phitmp.fix_mode(22,0);
	XERUS_LOG(info,phitmp.frob_norm());


	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift);
	for (size_t i = 0; i< test_number; ++i){
		builder.reset(sample);
		value_t val1 = builder.contract();
		auto ek = makeUnitVector(sample,d);
		value_t val2 = contract_TT(Hs,phi,ek);
		XERUS_LOG(info, "Sample = \t" << sample << std::setprecision(12) << " \t"<< std::abs(val1 - val2));
		sample = TrialSample(sample,d);
	}
	auto start = std::chrono::steady_clock::now();
	for (size_t i = 0; i< test_number2; ++i){
		builder.reset(sample);
		value_t val1 = builder.contract();
		sample = TrialSample(sample,d);
	}
	auto end = std::chrono::steady_clock::now();
	XERUS_LOG(info, "Elapsed time in seconds : "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000
		<< " sec");


}



TTTensor makeUnitVector(std::vector<size_t> sample, size_t d){
	std::vector<size_t> index(d, 0);
	for (size_t i : sample)
		if (i < d)
			index[i] = 1;
	auto unit = TTTensor::dirac(std::vector<size_t>(d,2),index);
	return unit;
}

void project(TTTensor &phi, size_t p, size_t d){
	Index i1,i2,j1,j2,k1,k2;
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
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
	phi.round(1e-4);
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	phi /= phi.frob_norm();
	XERUS_LOG(info,phi.ranks());
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
