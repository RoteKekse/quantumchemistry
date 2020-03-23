#include <xerus.h>
#include "../../classes/loading_tensors.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"
#include "../../classes/QMC/trialfunctions.cpp"

void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);
std::vector<size_t> makeRandomSample(size_t p,size_t d);


int main(){
	size_t nob = 16,num_elec = 6, rank = 6, iterations = 1e6;
	Index i1,j1,k1,i2,j2,k2;
	auto P = particleNumberOperator(nob);

	TTTensor test_direction = TTTensor::random(std::vector<size_t>(nob,2),std::vector<size_t>(nob-1,rank)),test_direction2,test_direction3,test_direction4;
	TTTensor tang_root = TTTensor::random(std::vector<size_t>(nob,2),std::vector<size_t>(nob-1,rank));
	TTTensor res_ex,res_ex2;
	test_direction /= test_direction.frob_norm();
	tang_root /= tang_root.frob_norm();

	XERUS_LOG(info, "Particle Number direction: " <<std::setprecision(16) << contract_TT(P,test_direction,test_direction));
	XERUS_LOG(info, "Particle Number root: " << std::setprecision(16) <<contract_TT(P,tang_root,tang_root));

	project(test_direction,num_elec,nob);
	test_direction.round(rank);
	test_direction /= test_direction.frob_norm();
	project(test_direction,num_elec,nob);
	test_direction.round(rank);
	test_direction /= test_direction.frob_norm();
	test_direction2 = test_direction;
	test_direction3 = test_direction;



	XERUS_LOG(info, "Particle Number direction: " << std::setprecision(16) <<contract_TT(P,test_direction,test_direction));
	XERUS_LOG(info,test_direction.ranks());

	test_direction2.round(3);
	test_direction2 /= test_direction2.frob_norm();

	XERUS_LOG(info, "Particle Number direction: " << std::setprecision(16) <<contract_TT(P,test_direction2,test_direction2));
	XERUS_LOG(info,test_direction2.ranks());

	XERUS_LOG(info,(test_direction2-test_direction).frob_norm());


	Tensor U = Tensor::random({test_direction.ranks()[2],test_direction.ranks()[2]}),Q,R;
	(Q(i1,j1),R(j1,k1)) = QR(U(i1,k1));
	test_direction3.move_core(3);
	(test_direction3.component(3)(j1,i1,i2),R(k1,j1)) = QR(test_direction3.component(3)(k1,i1,i2));
	test_direction3/=test_direction3.frob_norm();

	test_direction3.component(2)(i2,j2,i1) = test_direction3.get_component(2)(i2,j2,j1) * Q(i1,j1);
	test_direction3.component(3)(i1,j2,i2) = test_direction3.get_component(3)(j1,j2,i2) * Q(i1,j1);
	XERUS_LOG(info,(test_direction3-test_direction).frob_norm());
	test_direction4 = test_direction3;
	XERUS_LOG(info, "Particle Number direction: " << std::setprecision(16) <<contract_TT(P,test_direction4,test_direction4));
	XERUS_LOG(info,test_direction3.ranks());
	test_direction4.round(3);
	test_direction4 /= test_direction4.frob_norm();

	XERUS_LOG(info, "Particle Number direction: " << std::setprecision(16) <<contract_TT(P,test_direction4,test_direction4));
	XERUS_LOG(info,test_direction4.ranks());
	XERUS_LOG(info,(test_direction3-test_direction4).frob_norm());

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
	for (size_t k = 0; k <= d; ++k){

		if (p != k){
			auto PNk = particleNumberOperator(k,d);
			PNk.move_core(0);
			phi(i1&0) = PNk(i1/2,k1/2) * phi (k1&0);
			value_t f = (value_t) p - (value_t) k;
			phi /=  f;
			//phi.round(1e-12);
			phi /= phi.frob_norm();

		}
	}
	phi /= phi.frob_norm();
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
