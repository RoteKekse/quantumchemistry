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


	std::valarray<size_t> linear;
	std::valarray<size_t> tree;
	for (size_t rank : {5,10,20,40,80,150}){
	TTTensor phi = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,rank));

		ContractPsiHek builder(phi,d,p,path_T,path_V,0.0, shift,hf);


		builder.reset(hf);
		tree.emplace_back(builder.preparePsiEval());
		linear.emplace_back(p*(d-p)*(3*d*p-2*d-3*p*p+4)/32*d*rank*rank);
	}
	XERUS_LOG(info, "tree " << tree);
	XERUS_LOG(info, "line " << linear);
	XERUS_LOG(info, "ratio " << tree/linear);


	return 0;
}
