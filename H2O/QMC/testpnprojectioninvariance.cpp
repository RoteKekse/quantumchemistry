#include <xerus.h>
#include "classes/tangentialOperation.cpp"
#include "classes/trialfunctions.cpp"

#include "../loading_tensors.cpp"

void project(TTTensor &phi, size_t p, size_t d);
TTOperator particleNumberOperator(size_t k, size_t d);
TTOperator particleNumberOperator(size_t d);
TTTensor makeUnitVector(std::vector<size_t> sample, size_t d);
std::vector<size_t> makeRandomSample(size_t p,size_t d);


int main(){
	size_t nob = 32,num_elec = 8, rank = 10, iterations = 1e6;

	auto P = particleNumberOperator(nob);

	TTTensor test_direction = TTTensor::random(std::vector<size_t>(nob,2),std::vector<size_t>(nob-1,rank));
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
//	project(test_direction,num_elec,nob);
//	test_direction.round(rank);
//	test_direction /= test_direction.frob_norm();
	XERUS_LOG(info, "Particle Number direction: " << std::setprecision(16) <<contract_TT(P,test_direction,test_direction));

//	test_direction.move_core(6);
//	TTTensor td_short(std::vector<size_t>(6,2));
//	for (size_t i = 0; i < 5;++i)
//		td_short.set_component(i,test_direction.get_component(i));
//	auto ttt = test_direction.get_component(6);
//	ttt.fix_mode(2,2);
//	ttt.reinterpret_dimensions({ttt.dimensions[0],ttt.dimensions[1],1});
//	td_short.set_component(5,ttt);
//	td_short/=td_short.frob_norm();
//	auto Pt = particleNumberOperator(6);
//	XERUS_LOG(info, "Particle Number td short: " << std::setprecision(16) <<contract_TT(Pt,td_short,td_short));

	XERUS_LOG(info,tang_root.ranks());

	project(tang_root,num_elec,nob);
	tang_root.round(rank);
	XERUS_LOG(info,tang_root.ranks());
	tang_root /= tang_root.frob_norm();
	project(tang_root,num_elec,nob);
	tang_root.round(rank);
	XERUS_LOG(info,tang_root.ranks());

	tang_root /= tang_root.frob_norm();
//	project(tang_root,num_elec,nob);
//	tang_root.round(rank);
//	tang_root /= tang_root.frob_norm();
	XERUS_LOG(info, "Particle Number root: " << std::setprecision(16) <<contract_TT(P,tang_root,tang_root));
	XERUS_LOG(info,tang_root.ranks());

	TangentialOperation top(tang_root);
	auto tang_ex = top.localProduct(test_direction,true);
	res_ex = top.builtTTTensor(tang_ex);

	XERUS_LOG(info,"Norm residual " << std::setprecision(16) <<res_ex.frob_norm());
	res_ex /= res_ex.frob_norm();
	XERUS_LOG(info, "Particle Number residual: " << std::setprecision(16) << contract_TT(P,res_ex,res_ex));

	Index i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3,k4;
	auto TT3 = tang_root.get_component(2);
	auto TT4 = tang_root.get_component(3);


//	Tensor U = Tensor::random({6,6}),Q,R;
//	(Q(i1,j1),R(j1,k1)) = QR(U(i1,k1));
//	Tensor Uinv = pseudo_inverse(U,1);
//	Tensor test;
//	test(i1,i2) = U(i1,j1)*Uinv(j1,i2);

//	TT3(i1,i2,i3) = TT3(i1,i2,k1)*U(k1,i3);
//	TT4(i1,i2,i3) = Uinv(i1,k1)*TT4(k1,i2,i3);
//	tang_root.set_component(2,TT3);
//	tang_root.set_component(3,TT4);


	size_t idx = 5;
	tang_root.move_core(idx);
	TTOperator Py = TTOperator(std::vector<size_t>(2*nob,2)),tto1,tto2;
	auto id2 = Tensor::identity({2,2});
	id2.reinterpret_dimensions({1,2,2,1});
	Py.set_component(idx,id2);
	for (size_t i = 0; i < idx; ++i){
		auto ti = tang_root.get_component(i);
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
	for (size_t i = idx+1; i < nob; ++i){
		auto ti = tang_root.get_component(i);
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

	for (size_t i =0; i < tang_root.get_component(2).dimensions[2];++i){
		Tensor tt({1});
		tt[0]=1;
		tt(i1,i2,i3,k4) = tt(k1)*tang_root.get_component(0)(k1,i1,k2) *tang_root.get_component(1)(k2,i2,k3) *tang_root.get_component(2)(k3,i3,k4);
		tt.fix_mode(3,i);
		tt.reinterpret_dimensions({1,8});
		XERUS_LOG(info,"\n" << tt);
	}

	Tensor ttt({1});
	ttt[0]=1;
	Tensor ttt2({1});
	ttt2[0]=1;
	ttt(i1,i2,i3,j1,j2,j3) = ttt(k1)*Py.get_component(0)(k1,i1,j1,k2) *Py.get_component(1)(k2,i2,j2,k3) *Py.get_component(2)(k3,i3,j3,k4) *ttt(k4);
	ttt2(i1,i2,i3,j1,j2,j3) = ttt2(k1)*Py.get_component(4)(k1,i1,j1,k2) *Py.get_component(5)(k2,i2,j2,k3) *Py.get_component(6)(k3,i3,j3,k4) *ttt2(k4);
  ttt.reinterpret_dimensions({8,8});
  ttt2.reinterpret_dimensions({8,8});
	XERUS_LOG(info,"\n" << ttt);
	XERUS_LOG(info,"\n" << ttt2);

	Tensor test1({1});
	auto unit1 = makeUnitVector({0,1,4,5},nob);
	auto unit2 = makeUnitVector({0,1,4},nob);
	test1() = unit1(i1&0)*Py(i1/2,j1/2)*unit1(j1&0);
	XERUS_LOG(info,test1[0]);
	test1() = unit2(i1&0)*Py(i1/2,j1/2)*unit2(j1&0);
	XERUS_LOG(info,test1[0]);
	test1() = unit1(i1&0)*Py(i1/2,j1/2)*unit2(j1&0);
	XERUS_LOG(info,test1[0]);
	auto unit_chop = unit1.chop(idx);
	auto u1 = Tensor(unit_chop.first);
	u1.reinterpret_dimensions({1,8})	;
	auto u2 = Tensor(unit_chop.second);
	u2.reinterpret_dimensions({1,8});
	XERUS_LOG(info,"\n" << u1);
	XERUS_LOG(info,"\n" << u2);

	auto unit2_chop = unit2.chop(idx);
	auto u3 = Tensor(unit2_chop.first);
	u3.reinterpret_dimensions({1,8})	;
	auto u4 = Tensor(unit2_chop.second);
	u4.reinterpret_dimensions({1,8});
	XERUS_LOG(info,"\n" << u3);
	XERUS_LOG(info,"\n" << u4);


//	Tensor test1({1});
//	std::vector<size_t> s1 = {1,3,5,7,8,10}, s2;
//	TTTensor test_dir2 = makeUnitVector(s1,nob);
////	for (size_t i = 0; i < rank-1; ++i){
////		s1 = TrialSample(s1,nob);
////		test_dir2+= makeUnitVector(s1,nob);
////	}
//	test_dir2+= makeUnitVector({1,3,5,6,7 ,10},nob);
//
//	test_dir2 /= test_dir2.frob_norm();
//	XERUS_LOG(info, "Particle Number test dir 2: " << std::setprecision(16) <<contract_TT(P,test_dir2,test_dir2));
//	//test1() = unit1(i1&0)*Py(i1/2,j1/2)*unit2(j1&0);
//
//	idx = 6;
//	test_dir2.move_core(idx);
//	TTOperator Py2 = TTOperator(std::vector<size_t>(2*nob,2));
//	id2 = Tensor::identity({2,2});
//	id2.reinterpret_dimensions({1,2,2,1});
//	Py2.set_component(idx,id2);
//	for (size_t i = 0; i < idx; ++i){
//		auto ti = test_dir2.get_component(i);
//		if (i == idx-1){
//			ti(i1,j1,i2,j2) = ti(i1,i2,k1) * ti(j1,j2,k1);
//			auto dim = ti.dimensions;
//			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],1});
//		}
//		else{
//			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
//			auto dim = ti.dimensions;
//			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
//		}
//		Py2.set_component(i,ti);
//	}
//	for (size_t i = idx+1; i < nob; ++i){
//		auto ti = test_dir2.get_component(i);
//		if (i == idx+1){
//			ti(i2,j2,i3,j3) = ti(k1,i2,i3) * ti(k1,j2,j3);
//			auto dim = ti.dimensions;
//			ti.reinterpret_dimensions({1,dim[0],dim[1],dim[2]*dim[3]});
//		}
//		else{
//			ti(i1,j1,i2,j2,i3,j3) = ti(i1,i2,i3) * ti(j1,j2,j3);
//			auto dim = ti.dimensions;
//			ti.reinterpret_dimensions({dim[0]*dim[1],dim[2],dim[3],dim[4]*dim[5]});
//		}
//		Py2.set_component(i,ti);
//	}
//
//
//	tto1(i1/2,j1/2) = P(i1/2,k1/2)*Py2(k1/2,j1/2);
//	tto2(i1/2,j1/2) = Py2(i1/2,k1/2)*P(k1/2,j1/2);
//
//	tto1.move_core(0);
//	tto2.move_core(0);
//	XERUS_LOG(info,(tto1).frob_norm());
//	XERUS_LOG(info,(tto2).frob_norm());
//	XERUS_LOG(info,(tto1-tto2).frob_norm());
//
//
//	s1 = {{1,3,5,7,8,10}};
//	s2 = {{1,3,5,7,10}};
//	auto unit1 = makeUnitVector(s1,nob);
//	auto unit2 = makeUnitVector(s2,nob);
//	test1() = unit1(i1&0)*Py2(i1/2,j1/2)*unit2(j1&0);
//	XERUS_LOG(info,test1[0]);
//	test1() = unit1(i1&0)*Py2(i1/2,j1/2)*unit1(j1&0);
//	XERUS_LOG(info,test1[0]);
//	test1() = unit2(i1&0)*Py2(i1/2,j1/2)*unit2(j1&0);
//	XERUS_LOG(info,test1[0]);
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
