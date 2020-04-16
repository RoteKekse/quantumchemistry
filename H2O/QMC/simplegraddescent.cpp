#include <xerus.h>

#include "../../classes/QMC/tangential_parallel.cpp"
#include "../../classes/QMC/basic.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

int main(){
	size_t nob = 24,num_elec = 8,iterations = 1e5,pos = 5, numIter = 10;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=1.5;
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };
	value_t nuc = -52.4190597253;
	value_t alpha = 0.01;

	XERUS_LOG(info,"Loading Start vector from disc");
	TTTensor start, phi,res;
	read_from_disc("../data/hf_gradient_48.tttensor",start);
	phi = makeUnitVector(sample,2*nob);

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);

	XERUS_LOG(info,"Round start vector to " << eps << " keepin gisng values bigger than " << eps/std::sqrt(2*nob-1));
	start/= start.frob_norm();
	start.round(eps);
	start/= start.frob_norm();
	phi -= alpha*start;
	phi/=phi.frob_norm();
	XERUS_LOG(info,phi.ranks());

	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,sample,phi);
	TangentialOperation top(phi);
	tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
	tang.uvP.xbase.second = top.xbasis[1];

	for (size_t i = 0; i < numIter;++i){
		ev_app = tang.get_eigenvalue();
		XERUS_LOG(info, "Eigenvalue approx. " << std::setprecision(8) << ev_app - shift +nuc);
		XERUS_LOG(info, "Eigenvalue exact   " << std::setprecision(8) << contract_TT(Hs,phi,phi)- shift +nuc);
		auto tang_app = tang.get_tangential_components(ev_app,0.005);
		res = top.builtTTTensor(tang_app);
		res /= res.frob_norm();

		phi -= alpha*res;

		phi /= phi.frob_norm();
		phi.round(eps);
		phi /= phi.frob_norm();
		XERUS_LOG(info,phi.ranks());

		//update
		top.update(phi);
		tang.update(phi);
		tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
		tang.uvP.xbase.second = top.xbasis[1];
	}











	return 0;
}
