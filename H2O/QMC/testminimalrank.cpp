#include <xerus.h>
#include "../loading_tensors.cpp"

using namespace xerus;
using xerus::misc::operator<<;


void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);
std::vector<size_t> makeRandomSample(size_t p,size_t d);

int main(){


	size_t d = 4, p = 2;
	Index i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3,k4;
	auto P = particleNumberOperator(d);

	auto unit1 = makeUnitVector({0,1},d);
	auto unit2 = 2*makeUnitVector({0,2},d);
	auto unit3 = 3*makeUnitVector({0,3},d);
	auto unit4 = 4*makeUnitVector({1,2},d);
	auto unit5 = 5*makeUnitVector({1,3},d);
	auto unit6 = 6*makeUnitVector({2,3},d);

	auto TT = unit2;
	//TT += unit2;
	TT += unit3;
	TT += unit4;
	TT += unit5;
	TT += unit6;
	TT.round(1e-12);
	TT /= TT.frob_norm();
	XERUS_LOG(info,TT.ranks());
	XERUS_LOG(info, "Particle Number TT: " <<std::setprecision(16) << contract_TT(P,TT,TT));

	auto TT1 = TT.get_component(0);
	auto TT2 = TT.get_component(1);
	auto TT3 = TT.get_component(2);
	auto TT4 = TT.get_component(3);

	Tensor result;

	Tensor U = Tensor::random({TT2.dimensions[2],TT2.dimensions[2]}),Q,R;
	(Q(i1,j1),R(j1,k1)) = QR(U(i1,k1));
	Tensor Uinv = pseudo_inverse(U,1);
	Tensor test;
	test(i1,i2) = U(i1,j1)*Uinv(j1,i2);

	TT2(i1,i2,i3) = TT2(i1,i2,k1)*U(k1,i3);
	TT3(i1,i2,i3) = Uinv(i1,k1)*TT3(k1,i2,i3);

	TT.set_component(1,TT2);
	TT.set_component(2,TT3);
	TT /= TT.frob_norm();
	XERUS_LOG(info, "Particle Number TT: " <<std::setprecision(16) << contract_TT(P,TT,TT));

	size_t idx = 0;
	TT1.fix_mode(0,0);
	TT2.fix_mode(2,idx);
	TT3.fix_mode(0,idx);
	TT4.fix_mode(2,0);

	result(i1,i2,i3,i4)= TT1(i1,k1) * TT2(k1,i2)*TT3(i3,k2)*TT4(k2,i4);

	XERUS_LOG(info,"[0,0,0,0]" <<result[{0,0,0,0}]);
	XERUS_LOG(info,"[0,0,0,1]" <<result[{0,0,0,1}]);
	XERUS_LOG(info,"[0,0,1,0]" <<result[{0,0,1,0}]);
	XERUS_LOG(info,"[0,0,1,1]" <<result[{0,0,1,1}]);
	XERUS_LOG(info,"[0,1,0,0]" <<result[{0,1,0,0}]);
	XERUS_LOG(info,"[0,1,0,1]" <<result[{0,1,0,1}]);
	XERUS_LOG(info,"[0,1,1,0]" <<result[{0,1,1,0}]);
	XERUS_LOG(info,"[0,1,1,1]" <<result[{0,1,1,1}]);
	XERUS_LOG(info,"[1,0,0,0]" <<result[{1,0,0,0}]);
	XERUS_LOG(info,"[1,0,0,1]" <<result[{1,0,0,1}]);
	XERUS_LOG(info,"[1,0,1,0]" <<result[{1,0,1,0}]);
	XERUS_LOG(info,"[1,0,1,1]" <<result[{1,0,1,1}]);
	XERUS_LOG(info,"[1,1,0,0]" <<result[{1,1,0,0}]);
	XERUS_LOG(info,"[1,1,0,1]" <<result[{1,1,0,1}]);
	XERUS_LOG(info,"[1,1,1,0]" <<result[{1,1,1,0}]);
	XERUS_LOG(info,"[1,1,1,1]" <<result[{1,1,1,1}]);


	idx = 2;
	TT.move_core(idx);
	TTOperator Py = TTOperator(std::vector<size_t>(2*d,2)),tto1,tto2;
	auto id2 = Tensor::identity({2,2});
	id2.reinterpret_dimensions({1,2,2,1});
	Py.set_component(idx,id2);
	for (size_t i = 0; i < idx; ++i){
		auto ti = TT.get_component(i);
		if (i == idx-1){
			ti(i1,j1,i2,j2) = ti(i1,i2,k1) * ti(j1,j2,k1);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],1});
		}
		else{
			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
		}
		Py.set_component(i,ti);
	}
	for (size_t i = idx+1; i < d; ++i){
		auto ti = TT.get_component(i);
		if (i == idx+1){
			ti(i2,j2,i3,j3) = ti(k1,i2,i3) * ti(k1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({1,dim[0],dim[1],dim[2]*dim[3]});
		}
		else{
			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
			auto dim = ti.dimensions;
			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
		}
		Py.set_component(i,ti);
	}

	tto1(i1/2,j1/2) = P(i1/2,k1/2)*Py(k1/2,j1/2);
	tto2(i1/2,j1/2) = Py(i1/2,k1/2)*P(k1/2,j1/2);

	tto1.move_core(0);
	tto2.move_core(0);
	XERUS_LOG(info,(tto1).frob_norm());
	XERUS_LOG(info,(tto2).frob_norm());
	XERUS_LOG(info,(tto1-tto2).frob_norm());

	XERUS_LOG(info,TT.get_component(2).dimensions);

	for (size_t i =0; i < TT.get_component(1).dimensions[2];++i){
		Tensor tt({1});
		tt[0]=1;
		tt(i1,i2,k3) = tt(k1)*TT.get_component(0)(k1,i1,k2) *TT.get_component(1)(k2,i2,k3);
		tt.fix_mode(2,i);
		tt.reinterpret_dimensions({1,4});
		XERUS_LOG(info, tt);
	}

	Tensor ttt({1});
	ttt[0]=1;
	Tensor ttt2({1});
	ttt2[0]=1;
	XERUS_LOG(info,ttt.dimensions);
	XERUS_LOG(info,Py.get_component(0).dimensions);
	XERUS_LOG(info,Py.get_component(1).dimensions);
	ttt(i1,i2,j1,j2) = ttt(k1)*Py.get_component(0)(k1,i1,j1,k2) *Py.get_component(1)(k2,i2,j2,k3)  *ttt(k3);
	XERUS_LOG(info,"Hello");
	ttt2(i1,j1) = ttt2(k1)*Py.get_component(3)(k1,i1,j1,k2) *ttt2(k2);
	XERUS_LOG(info,"Hello");
	ttt.reinterpret_dimensions({4,4});
	XERUS_LOG(info,"Hello");
	ttt2.reinterpret_dimensions({2,2});
	XERUS_LOG(info,"\n" << ttt);
	XERUS_LOG(info,"\n" << ttt2);

	return 0;
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
