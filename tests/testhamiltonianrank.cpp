#include <xerus.h>
#include <../classes/helpers.cpp>

using namespace xerus;
using xerus::misc::operator<<;


int main(){

	for (size_t d = 8; d < 20; d=d+2){
		XERUS_LOG(info,d);
		size_t idx = d/2;

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(1.0,1.0);
		TTOperator res(std::vector<size_t>(2*d,2));
		for (size_t i = 0; i < idx; ++i){
			for (size_t j = 0; j < idx; ++j){
				for (size_t k = idx/2; k < d; ++k){
					for (size_t l = idx/2; l < d; ++l){
						double number = distribution(generator);
						res += number*return_two_e_ac(i,j,k,l,d);


					}
				}
			}
		}
		XERUS_LOG(info,res.ranks());
		res.round(1e-10);
		XERUS_LOG(info,res.ranks());
	}





	return 0;
}
