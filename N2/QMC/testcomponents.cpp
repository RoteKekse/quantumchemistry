#include <xerus.h>

#include "../../classes/QMC/tangential_parallel.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

TTTensor buildStartVector(std::vector<size_t> sample, size_t p, size_t d);

int main(){
	size_t nob = 60,num_elec = 14,iterations = 1e5;
	value_t ev, shift = 25.0, ev_app;
	std::string molecule = "N2";
	std::vector<size_t> hf = { 0, 1,2,3,4,5,6,7,8,9,10,11,12,13 };

	xerus::TTTensor phi,res,res_last;
	phi = buildStartVector(hf,num_elec,2*nob);
	phi /= phi.frob_norm(); //normalize
	project(phi,num_elec,2*nob);

	std::string path_T = "../data/T_N2_120.tensor";
	std::string path_V= "../data/V_N2_120.tensor";
	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,hf,phi);
	phi.move_core(0,true);
	tang.uvP.xbase.first = phi;
	phi.move_core(2*nob-1,true);
	tang.uvP.xbase.second = phi;


	ev_app = tang.get_eigenvalue();
	XERUS_LOG(info, "Eigenvalue approx. " << ev_app);

	auto tang_app = tang.get_tangential_components(ev_app,0.005);


	res = tang.builtTTTensor(tang_app);
	std::string name2 = "../data/residual_app_"+ molecule + "_" + std::to_string(2*nob) +".tttensor";
	write_to_disc(name2,res);






	return 0;
}


TTTensor buildStartVector(std::vector<size_t> sample, size_t p, size_t d){

	return makeUnitVector(sample,d);
}
