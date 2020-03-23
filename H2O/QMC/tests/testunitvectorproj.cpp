#include <xerus.h>

#include "../classes_old/tangentialOperation.cpp"
#include "../classes_old/unitvectorprojection.cpp"
#include "../loading_tensors.cpp"

TTOperator particleNumberOperator(size_t k, size_t d);
void project(TTTensor &phi, size_t p, size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);


int main(){
	size_t nob = 24,num_elec = 8,iterations = 1e5,pos = 5;
	XERUS_LOG(info,"Test unitvector projection");


	xerus::TTTensor phi,res,res_last;
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));
	read_from_disc("../data/eigenvector_H2O_48_3_-23.647510_benchmark.tttensor",phi);
	phi /= phi.frob_norm(); //normalize
	project(phi,num_elec,2*nob);

	TangentialOperation top(phi);
	std::vector<size_t> test_sample = { 0, 1, 4, 5, 22, 23, 30, 31 };
	auto unit = makeUnitVector(test_sample, 2*nob);
	XERUS_LOG(info,unit.frob_norm());
	auto tang_test = top.localProduct(unit);

	unitVectorProjection uvP(phi);
	auto loc_grad = uvP.localProduct(test_sample,pos,true);

	TTTensor tang_ex_TT(std::vector<size_t>(2*nob,2)), tang_app_TT(std::vector<size_t>(2*nob,2));
	for (size_t i = 0; i < pos; ++i){
		tang_ex_TT.set_component(i,top.xbasis[1].get_component(i));
		tang_app_TT.set_component(i,uvP.xbase.second.get_component(i));
	}
	tang_ex_TT.set_component(pos,tang_test[pos]);
	tang_app_TT.set_component(pos,loc_grad);
	for (size_t i = pos+1; i < 2*nob; ++i){
		tang_ex_TT.set_component(i,top.xbasis[0].get_component(i));
		tang_app_TT.set_component(i,uvP.xbase.first.get_component(i));
	}
	tang_ex_TT.move_core(0);
	tang_app_TT.move_core(0);
	XERUS_LOG(info,"Error:  "<< (tang_ex_TT-tang_app_TT).frob_norm());





	return 0;
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
