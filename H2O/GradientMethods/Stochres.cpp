#include <xerus.h>
#include <experimental/random>
using xerus::misc::operator<<;


using namespace xerus;


std::vector<Tensor> get_random_unit_vector(size_t d, size_t particles){
	std::vector<Tensor>  unit(d,Tensor({2}));
	Tensor unit1 = Tensor::dirac({2},{0});
	Tensor unit2 = Tensor::dirac({2},{1});
  srand( (unsigned)time( NULL ) );
  size_t i = 0;
  for (i = 0; i < d; ++i)
  	unit[i] = unit1;
  i = 0;
  std::vector<int> visited;
	while ( i < particles){
		int r = std::experimental::randint(0, 49);
		if(std::find(visited.begin(), visited.end(), r) == visited.end()){
			visited.emplace_back(r);
			unit[r] = unit2;
			++i;
		}
	}

	return unit;
}

void contract_TT(const TTOperator& A,const std::vector<Tensor>& x, TTTensor& result){
	size_t d = A.order()/2;
	Tensor tmp;
	for (size_t i = 0; i < d; ++i){
		Index i1,i2,i3,j1,j2;
		auto Ai = A.get_component(i);
		auto uvec = x[i];
		tmp(i1,i2,i3) = Ai(i1,i2,j1,i3)*uvec(j1);
		result.set_component(i,tmp);
	}
}

value_t contract_TT(const TTTensor& y,const std::vector<Tensor>& x){
	size_t d = y.order();
	Tensor tmp = Tensor::ones({1});
	for (size_t i = 0; i < d; ++i){
		Index i1,i2,i3,j1,j2,j3;
		auto yi = y.get_component(i);
		auto uvec = x[i];
		tmp(i1) = tmp(j1) * yi(j1,j2,i1)*uvec(j2);
	}
	return tmp[0];
}

void getRes_Stoch(const TTOperator& A,const TTTensor& x, const TTOperator M, double lambda, TTTensor& res, const size_t sample_size, const size_t maxRank){
	size_t d = A.order()/2;
	Index ii,jj,kk,ll;
	value_t xv;
	TTTensor Av(std::vector<size_t>(d,2));
	TTTensor Mv(std::vector<size_t>(d,2));
	res = TTTensor(std::vector<size_t>(d,2));
	for (size_t iter = 0; iter < sample_size; ++iter){
		XERUS_LOG(info, iter);
		std::vector<Tensor> vec = get_random_unit_vector(d, 10);
		xv = contract_TT(x,vec);
		contract_TT(A,vec,Av);
		XERUS_LOG(info, "hello");

		contract_TT(M,vec,Mv);
		XERUS_LOG(info, "hello");

//		Av.round(maxRank);
//		Mv.round(maxRank);

		res += Av - xv * Mv;
		XERUS_LOG(info, res.ranks());

		res.round(maxRank);


	}

}
