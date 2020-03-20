#include <xerus.h>
using namespace xerus;
using xerus::misc::operator<<;

int main(){

	size_t d = 50;
	std::vector<size_t> partial = {0,10,20,30,40,50};
	Index ii,jj,kk,i1,i2,i3,j1,j2,j3;
	XERUS_LOG(info, "---- Load Partial Ops ----");
	std::vector<TTOperator> ttolist;
	ttolist.reserve(partial.size());
	for (size_t i = 0; i < partial.size();++i){
		std::string name = "../data/hamiltonian_H2O_50_partial_"+ std::to_string(partial[i]) + ".ttoperator";
		std::ifstream read(name.c_str());
		misc::stream_reader(read,ttolist[i],xerus::misc::FileFormat::BINARY);
		read.close();
	}

	XERUS_LOG(info, "Loading Hamiltonian Operator");
	xerus::TTOperator H;
	std::string name = "../data/hamiltonian_H2O_" + std::to_string(d) +"_full_3.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,H,xerus::misc::FileFormat::BINARY);
	read.close();
	XERUS_LOG(info, "Loading Hamiltonian Operator -- Finished");

	TTTensor slater = TTTensor::dirac(std::vector<size_t>(d,2),1125899906842624 - 1099511627776);

	TTTensor tmp,tmp2;
	XERUS_LOG(info, "Norm of full Hamiltonian: " << (H).frob_norm());
	XERUS_LOG(ingo,H.ranks());
	tmp () = slater(ii&0) * H(ii/2,jj/2)* slater(jj&0);
//	XERUS_LOG(info, "The expectation for the exact Hamiltonian is   " << tmp[0]);
//	for (size_t i = 0; i < partial.size();++i){
//		Tensor tmp2;
//		XERUS_LOG(info, "For partial " << partial[i] << " the difference is: " << (H-ttolist[i]).frob_norm());
//		XERUS_LOG(ingo,ttolist[i].ranks());
//		Tensor left = Tensor::dirac({1,1,1},0);
//		Tensor right = Tensor::dirac({1,1,1},0);
//		for (size_t j = 0; j < d;j++)
//			left(i1,i2,i3) = left(j1,j2,j3) *slater.get_component(j)(j1,ii,i1)* ttolist[i].get_component(j)(j2,ii,jj,i2)  * slater.get_component(j)(j3,jj,i3);
//		tmp2 () = left(i1,i2,i3) * right(i1,i2,i3);
//		XERUS_LOG(info, "The expectation for the partial Hamiltonian is " << tmp2[0]);
//
//	}
	tmp (ii&0) =  H(ii/2,jj/2)* slater(jj&0);
	auto lastH = ttolist[5];
	tmp2 (ii&0) =  lastH(ii/2,jj/2)* slater(jj&0);
	XERUS_LOG(info,"Diff = " << (H-lastH).frob_norm());
	XERUS_LOG(info,"Diff = " << (tmp-tmp2).frob_norm());




	return 0;
}
