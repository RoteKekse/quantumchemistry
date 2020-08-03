#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){

	for (size_t d = 8; d < 28; d=d+2){
		XERUS_LOG(info,d);
		size_t idx = d/2;

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(1.0,1.0);
		TTOperator res(std::vector<size_t>(2*d,2));
		TTOperator res2(std::vector<size_t>(2*d,2));
		for (size_t i = 0; i < idx; ++i){
			for (size_t j = idx; j < d; ++j){
				for (size_t k = idx; k < d; ++k){
					for (size_t l = idx; l < d; ++l){
						double number = distribution(generator);
						res += number*return_two_e_ac(i,j,k,l,d);
					}
				}
			}
		}
		for (size_t i = 0; i < idx; ++i){
			for (size_t j = 0; j < idx; ++j){
				for (size_t k = idx; k < d; ++k){
					for (size_t l = idx; l < d; ++l){
						double number = distribution(generator);
						res2 += number*return_two_e_ac(i,j,k,l,d);
					}
				}
			}
		}
//		for (size_t k = 0; k < idx; ++k){
//			for (size_t l = idx; l < d; ++l){
//				double number = distribution(generator);
//				//double number = 1;
//				res2 += number*return_one_e_ac(k,l,d);
//
//
//			}
//		}
		res.round(1e-9);
		res2.round(1e-8);
		XERUS_LOG(info,res.ranks());
		XERUS_LOG(info,res2.ranks());
	}





	return 0;
}
