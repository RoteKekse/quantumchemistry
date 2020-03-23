#include <xerus.h>

#include "../classes_old/tangential.cpp"

TTOperator particleNumberOperator(size_t k, size_t d);


int main(){
	srand(time(NULL));
	size_t d = 48, p = 8, iterations = 1e6;
	std::string path_T = "../data/T_H2O_48_bench.tensor";
	std::string path_V= "../data/V_H2O_48_bench.tensor";
	value_t nuc = -52.4190597253, ref = -76.25663396, shift = 25.0, f;
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };


	TTTensor phi;
	read_from_disc("../data/eigenvector_H2O_48_5_-23.700175_benchmark.tttensor",phi);
	Index i1,i2,j1,j2,k1,k2;
	for (size_t k = 0; k <= d; ++k){
		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			f = (value_t)p - (value_t) k;
			phi /=  f;
			phi.round(1e-12);
		}
	}
	phi.round(1e-6);
	XERUS_LOG(info, "Phi norm           " << phi.frob_norm());
	XERUS_LOG(info,phi.ranks());


	Tangential tang(d,p,iterations,path_T,path_V,shift,sample,phi);

	value_t ev = tang.get_eigenvalue();
	auto result = tang.get_tangential_components2(ev,0.00001);
	XERUS_LOG(info,"Eigenvalue = " << ev);


	return 0;
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
