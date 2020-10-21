#include "../classes/QMC/contractpsihek.cpp"

#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){
	std::string path_T = "../H2O/data/T_H2O_48_bench_single.tensor";
	std::string path_V = "../H2O/data/V_H2O_48_bench_single.tensor";
	size_t d = 48, p = 8;
	value_t shift = 25;
	std::vector<size_t> hf = {0,1,2,3,22,23,30,31};
	auto hf_idx = std::vector<size_t>(d,0);
	for (size_t i : hf)
		hf_idx[i] = 1;

	//TTTensor phi = TTTensor::dirac(std::vector<size_t>(d,2),hf_idx);
	TTTensor phi = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,5));

	ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift,hf);

	builder.reset(hf);
	builder.reset_psi(phi);
	builder.preparePsiEval_linear();
	XERUS_LOG(info, "linear " << builder.contract_tree());

	builder.reset(hf);
	builder.preparePsiEval();
	XERUS_LOG(info, "tree " << builder.contract_tree());


//	std::vector<size_t> linear1;
//	std::vector<size_t> linear2;
//	std::vector<size_t> tree;
//	for (size_t rank : {5,10,20,40,80,150,300,600}){
//		TTTensor phi = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,rank));
//
//		ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift,hf);
//
//
//		builder.reset(hf);
//		tree.emplace_back(builder.preparePsiEval());
//		linear1.emplace_back(p*(d-p)*(3*d*p-2*d-3*p*p+4)/32*d*rank*rank);
//		linear2.emplace_back(builder.preparePsiEval_linear());
//	}
//	XERUS_LOG(info, "tree " << tree);
//	XERUS_LOG(info, "line1 " << linear1);
//	XERUS_LOG(info, "line2 " << linear2);
//	for (size_t i = 0; i < tree.size(); ++i)
//		XERUS_LOG(info, value_t (tree[i]) / value_t (linear1[i]));


	return 0;
}
