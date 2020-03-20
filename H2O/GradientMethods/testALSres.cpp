#include <xerus.h>
#include "ALSres.cpp"

using namespace xerus;
using xerus::misc::operator<<;


int main(){
	size_t dim = 100;

	size_t rankx1 = 10;
	size_t rankx2 = 15;

	TTOperator id = xerus::TTOperator::identity(std::vector<size_t>(2*dim,2));

	TTTensor x_rounded = TTTensor::random(std::vector<size_t>(dim,2),std::vector<size_t>(dim-1,rankx1 ));
	TTTensor x = TTTensor::random(std::vector<size_t>(dim,2),std::vector<size_t>(dim-	1,rankx2 ));
	x_rounded/=x_rounded.frob_norm();
	x/=x.frob_norm();

	XERUS_LOG(info, "error before ALS   = " << (x-x_rounded).frob_norm());

	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS1   = " << (x-x_rounded).frob_norm());
	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS2   = " << (x-x_rounded).frob_norm());
	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS3   = " << (x-x_rounded).frob_norm());
	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS4   = " << (x-x_rounded).frob_norm());
	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS5   = " << (x-x_rounded).frob_norm());
	getRes(id,x,id,0,x_rounded);
	XERUS_LOG(info, "error after ALS6   = " << (x-x_rounded).frob_norm());

	TTTensor tmp = x;
	tmp.round(rankx1);
  XERUS_LOG(info, "error after round = " << (x-tmp).frob_norm());

  return 0;
}
