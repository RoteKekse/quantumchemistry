#include <xerus.h>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"

using namespace xerus;
using xerus::misc::operator<<;

TTOperator tang_proj(TTTensor &tang_root, size_t pos);


/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for Projection of Hartree Fock Solution ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");
	size_t nob = 60,num_elec = 14,iterations = 1e5;
	value_t ev, shift = 135.0, ev_app;
	std::string molecule = "N2";
	std::vector<size_t> hf = { 0, 1,2,3,4,5,6,7,8,9,10,11,12,13 };
	value_t nuc = 23.5724393955273;
	size_t pos = 10;

	auto eHF = makeUnitVector(hf,2*nob);
	TangentialOperation tang(eHF);
	auto Pop = tang_proj(eHF,pos);
	XERUS_LOG(info, Pop.frob_norm());

	return 0;
}

TTOperator tang_proj(TTTensor &tang_root, size_t pos){
	size_t d = tang_root.order();
	tang_root.move_core(pos);
	Index i1,i2,i3,j1,j2,j3,k1,k2;
	TTOperator Py = TTOperator(std::vector<size_t>(2*d,2)),tto1,tto2;
	auto id2 = Tensor::identity({2,2});
	id2.reinterpret_dimensions({1,2,2,1});
	Py.set_component(pos,id2);
	for (size_t i = 0; i < pos; ++i){
		auto ti = tang_root.get_component(i);
		if (i == pos-1){
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
	for (size_t i = pos+1; i < d; ++i){
		auto ti = tang_root.get_component(i);
		if (i == pos+1){
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
	return Py;
}



