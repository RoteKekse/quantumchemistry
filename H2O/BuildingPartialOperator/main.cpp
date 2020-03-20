#include <xerus.h>
#include "buildingpartialoperatorL2R.cpp"


int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");

	size_t d = 116;
	std::vector<size_t> partial = {54,60,66};
	std::string path_T = "../data/T_H2O_116.tensor";
	std::string path_V= "../data/V_H2O_116.tensor";
	value_t nuc = 8.80146457125193;
	BuildingPartialOperatorL2R builder(d,partial,path_T,path_V,nuc);



	XERUS_LOG(test,"Testing build");
	builder.build();

	XERUS_LOG(test,"Load old operator");
//	xerus::TTOperator op;
//	std::string name = "../hamiltonian_CH2_26_full.ttoperator";
//	std::ifstream read(name.c_str());
//	misc::stream_reader(read,op,xerus::misc::FileFormat::BINARY);
//	read.close();
//	XERUS_LOG(test, "ranks of operator: " << op.ranks());
	//XERUS_LOG(test, "frob norm diff: " << (op-builder.H).frob_norm());

	XERUS_LOG(info, "---- Save Test Result ----");
	for (size_t i = 0; i < partial.size();++i){
		std::string name = "../data/hamiltonian_H2O_"+std::to_string(d)+"_partial2_"+ std::to_string(partial[i]) + ".ttoperator";
		std::ofstream write(name.c_str());
		misc::stream_writer(write,builder.Hlist[i],xerus::misc::FileFormat::BINARY);
		write.close();
	}


//	XERUS_LOG(test,"Testing correctnes of operator comparuing with brute force ");
//
//	XERUS_LOG(info, "Loading Hamiltonian Operator");
//	xerus::TTOperator H;
//	std::string name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
//	std::ifstream read(name.c_str());
//	misc::stream_reader(read,H,xerus::misc::FileFormat::BINARY);
//	read.close();
//	XERUS_LOG(info, "Loading Hamiltonian Operator -- Finished");
//	//auto brute = builder.build_brute(10);
//	//XERUS_LOG(test, brute.ranks());
//	XERUS_LOG(test, builder.H.ranks());
//	XERUS_LOG(test, "frob norm diff (rel): " << (H-builder.H).frob_norm()/(builder.H).frob_norm());
//	XERUS_LOG(test, "frob norm diff:       " << (H-builder.H).frob_norm());

	return 0;
}
