#include <xerus.h>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>


using namespace xerus;
using xerus::misc::operator<<;



class Data_CH2 {
	public:

		const size_t d;
		size_t count_L;
		size_t count_R;
		TTOperator S_L;
		TTOperator S_R;
		std::vector<std::vector<TTOperator>> P_L;
		std::vector<std::vector<TTOperator>> P_R;
		std::vector<std::vector<TTOperator>> Q_L;
		std::vector<std::vector<TTOperator>> Q_R;
		Tensor H;
		Tensor V;
		Tensor N;
		TTOperator H_L;
		TTOperator H_R;
		std::string path;


	Data_CH2(const size_t _d)
			: d(_d), count_L(0), count_R(_d-1), path("../FCIDUMP.ch2_13")
		{
			H = make_H_CH2();
		  V = make_V_CH2();
		  N = make_Nuc_CH2();
		  S_L = TTOperator();
		  S_R = TTOperator();
		  P_L = initialize_P();
		  P_R = initialize_P();
		  Q_L = initialize_Q(true);
		  Q_R = initialize_Q(false);
		  H_L = TTOperator();
		  H_R = TTOperator();
		}

	// updates all stack recursively, increases count_L and decreases count_R
	void update(){
		count_L++;
		count_R--;
		update_PQ();
		update_PQ(false);
	}

	// updates each P_pq based on coming from left or right, note that a_k*a_k is 0
	void update_Ppq(size_t p, size_t q, bool left){
		size_t k = left ? count_L : count_R;
		size_t dim = left ? count_L + 1: d - count_R;
		TTOperator tmp = w(p,q,left ? 0 : count_R + 1,k) * return_annil_P(left ? 0 : 1,left ? dim - 1 : 0,dim);
		if (k > 1 && k < d - 2){
			for (size_t r = left ? 1 : k + 2; r < left ? k : d; ++r){
				tmp += w(p,q,r,k) * return_annil_P(left ? r : r - k,left ? dim - 1 : 0,dim);
			}
			if (left)
				tmp += add_identity(P_L[p][q]);
			else
				tmp += add_identity(P_R[p][q], true);
		}
		if(left)
			P_L[p][q] = tmp;
		else
			P_R[p][q] = tmp;
	}

	// updates each Q_pq based on coming from left or right,
	void update_Qpq(size_t p, size_t q, bool left){
		size_t k = left ? count_L : count_R;
		size_t dim = left ? count_L + 1: d - count_R;
		size_t pos_k = left ? dim - 1 : 0;
		TTOperator tmp = y(p,left ? 0 : count_R + 1,q,k) * return_ca(left ? 0 : 1,pos_k,dim);
		for (size_t r = left ? 1 : k + 2; r < left ? k : d; ++r){
			tmp += y(p,r,q,k) * return_annil_P(left ? r : r - k,pos_k,dim);
		}
		tmp += w(p,k,q,k) *return_ca(pos_k,pos_k,dim);
		if (left){
			tmp += add_identity(Q_L[p][q]);
			Q_L[p][q] = tmp;
		} else {
			tmp += add_identity(Q_R[p][q], true);
			Q_R[p][q] = tmp;
		}
	}

	// updates each elements in P_L or P_R depending on left = true or false
	// increases count ++
	void update_PQ(bool left=true){ //TODO update count_L and count_R before calling update functions
		for (size_t p = left ? count_L + 1 : 0; p < left ? d : count_R; ++p){
			for (size_t q = left ? count_L + 1 : 0; q < left ? d : count_R; ++q){
				update_Ppq(p,q,left);
				update_Qpq(p,q,left);
			}
		}
	}

	value_t w(size_t p, size_t q, size_t r, size_t s){
		return V[{p,q,r,s}] - V[{p,q,s,r}];
	}

	value_t y(size_t p, size_t r, size_t q, size_t s){
		return w(p,r,q,s) - w(p,s,q,r);
	}

	// initializes the d^2 array P_L and P_R with zero degree one Tensor Train Operators
	std::vector<std::vector<TTOperator>> initialize_P(){
		std::vector<TTOperator> tmp(d, TTOperator({2,2}));
		std::vector<std::vector<TTOperator>> result(d,tmp);
		return result;
	}

	std::vector<std::vector<TTOperator>> initialize_Q(bool left=true){
		std::vector<TTOperator> tmp(d, TTOperator({2,2}));
		std::vector<std::vector<TTOperator>> result(d,tmp);
		size_t k = left ? count_L : count_R;
		for (size_t p = 0; p < d; ++p){
			for (size_t q = 0; q < d; ++q){
				result[p][q] = w(p,k,q,k) *return_ca(0,0,1);
			}
		}
		return result;
	}

	// Takes the given operator and adds an identity to the right or left
	TTOperator add_identity(TTOperator A, bool left=false){
		count_L++;
		TTOperator tmpA(A.degree() + 1);
		for (size_t i = left ? 0 : 1; i < A.degree() + (left ? 0 : 1); ++i){
			tmpA.set_component(i,A.get_component(i)); //TODO check if component works instead of get_component
		}
		Tensor ident = Tensor::identity({1,2,2,1});
		tmpA.set_component(left ? A.degree() : 0,ident);
		return tmpA;
	}




	xerus::Tensor make_Nuc_CH2(){
		auto Nuc = xerus::Tensor({1});
		std::string line;
		std::ifstream input;
		input.open (path);
		size_t count = 0;
		while ( std::getline (input,line) )
		{
			count++;
			if (count > 4){
				std::vector<std::string> l;
				boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
				if (std::stoi(l[1]) == 0 && std::stoi(l[3]) == 0){
					Nuc[{0}] = stod(l[0]);
				}
			}
		}
		input.close();
		return Nuc;
	}

	xerus::Tensor make_H_CH2(){
		auto nob = d/2;
		auto H = xerus::Tensor({2*nob,2*nob});
		auto H_tmp = xerus::Tensor({nob,nob});
		std::string line;
		std::ifstream input;
		input.open (path);
		size_t count = 0;
		while ( std::getline (input,line) )
		{
			count++;
			if (count > 4){
				std::vector<std::string> l;
				boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
				if (std::stoi(l[1]) != 0 && std::stoi(l[3]) == 0){
					H_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1}] = stod(l[0]);
				}
			}
		}
		input.close();
		for (size_t i = 0; i < nob; i++){
			for (size_t j = 0; j < nob; j++){
				auto val = H_tmp[{i,j}];
				H[{2*i,2*j}] = val;
				H[{2*j,2*i}] = val;
				H[{2*i+1,2*j+1}] = val;
				H[{2*j+1,2*i+1}] = val;
			}
		}
		return H;
	}


	xerus::Tensor make_V_CH2(){
		auto nob = d/2;
		auto V = xerus::Tensor({2*nob,2*nob,2*nob,2*nob});
		auto V_tmp = xerus::Tensor({nob,nob,nob,nob});
		auto V_tmp2 = xerus::Tensor({nob,nob,nob,nob});
		std::string line;
		std::ifstream input;
		input.open (path);
		size_t count = 0;
		while ( std::getline (input,line) )
		{
			count++;
			if (count > 4){
				std::vector<std::string> l;
				boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
				if (std::stoi(l[1]) != 0 && std::stoi(l[3]) != 0){
					V_tmp[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1,static_cast<size_t>(std::stoi(l[3]))-1,static_cast<size_t>(std::stoi(l[4]))-1}] = stod(l[0]);
				}
			}
		}
		input.close();
		for (size_t i = 0; i < nob; i++){
				for (size_t j = 0; j <= i; j++){
					for (size_t k = 0; k<= i; k++){
						for (size_t l = 0; l <= (i==k ? j : k); l++){
							auto value = V_tmp[{i,j,k,l}];
							V_tmp2[{i,k,j,l}] = value;
							V_tmp2[{j,k,i,l}] = value;
							V_tmp2[{i,l,j,k}] = value;
							V_tmp2[{j,l,i,k}] = value;
							V_tmp2[{k,i,l,j}] = value;
							V_tmp2[{l,i,k,j}] = value;
							V_tmp2[{k,j,l,i}] = value;
							V_tmp2[{l,j,k,i}] = value;
						}
					}
				}
		}
		for (size_t i = 0; i < nob; i++){
					for (size_t j = 0; j < nob; j++){
						for (size_t k = 0; k < nob; k++){
							for (size_t l = 0; l < nob; l++){
								auto value = V_tmp2[{i,j,k,l}];
								V[{2*i,2*j,2*k,2*l}] = value;
								V[{2*i+1,2*j,2*k+1,2*l}] = value;
								V[{2*i,2*j+1,2*k,2*l+1}] = value;
								V[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
							}
						}
					}
		}

		return V;
	}

	//Creation of Operators
	xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this
		xerus::Index i1,i2,jj, kk, ll;
		auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));
		auto id = xerus::Tensor({2,2});
		id[{0,0}] = 1.0;
		id[{1,1}] = 1.0;
		auto s = xerus::Tensor({2,2});
		s[{0,0}] = 1.0;
		s[{1,1}] = -1.0;
		auto annhil = xerus::Tensor({2,2});
		annhil[{0,1}] = 1.0;
		for (size_t m = 0; m < d; ++m){
			auto tmp = m < i ? s : (m == i ? annhil : id );
			auto res = xerus::Tensor({1,2,2,1});
			res[{0,0,0,0}] = tmp[{0,0}];
			res[{0,1,1,0}] = tmp[{1,1}];
			res[{0,1,0,0}] = tmp[{1,0}];
			res[{0,0,1,0}] = tmp[{0,1}];
			a_op.set_component(m, res);
		}
		return a_op;
	}

	xerus::TTOperator return_annil_P(size_t i, size_t j, size_t d){ // TODO write tests for this
		auto an1 = return_annil(i,d);
		auto an2 = return_annil(j,d);
		xerus::TTOperator res;
		xerus::Index ii,jj,kk;
		res(ii/2,jj/2) = an1(ii/2,kk/2) * an2(kk/2, jj/2);
		return res;
	}


	xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
		xerus::Index i1,i2,jj, kk, ll;
		auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

		auto id = xerus::Tensor({2,2});
		id[{0,0}] = 1.0;
		id[{1,1}] = 1.0;
		auto s = xerus::Tensor({2,2});
		s[{0,0}] = 1.0;
		s[{1,1}] = -1.0;
		auto create = xerus::Tensor({2,2});
		create[{1,0}] = 1.0;
		for (size_t m = 0; m < d; ++m){
			auto tmp = m < i ? s : (m == i ? create : id );
			auto res = xerus::Tensor({1,2,2,1});
			res[{0,0,0,0}] = tmp[{0,0}];
			res[{0,1,1,0}] = tmp[{1,1}];
			res[{0,1,0,0}] = tmp[{1,0}];
			res[{0,0,1,0}] = tmp[{0,1}];
			c_op.set_component(m, res);
		}
		return c_op;
	}

	xerus::TTOperator return_ca(size_t i, size_t j, size_t d){ // TODO write tests for this
		auto cr = return_create(i,d);
		auto an = return_annil(j,d);
		xerus::TTOperator res;
		xerus::Index ii,jj,kk;
		res(ii/2,jj/2) = cr(ii/2,kk/2) * an(kk/2, jj/2);
		return res;
	}

	xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d){ //todo test
		auto cr1 = return_create(i,d);
		auto cr2 = return_create(j,d);
		auto an1 = return_annil(k,d);
		auto an2 = return_annil(l,d);
		xerus::TTOperator res;
		xerus::Index ii,jj,kk,ll,mm;
		res(ii/2,mm/2) = cr1(ii/2,jj/2) * cr2(jj/2,kk/2) * an1(kk/2,ll/2) * an2(ll/2,mm/2);
		return res;
	}

	xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim){
		auto res = xerus::TTOperator(std::vector<size_t>(2*dim,2));
		if (a!=b && c!=d)
			res += return_two_e_ac(a,b,d,c,dim);
			res += return_two_e_ac(a+1,b+1,d+1,c+1,dim);
		if (c != d)
			res += return_two_e_ac(a+1,b,d,c+1,dim);
		res += return_two_e_ac(a,b+1,d+1,c,dim);


		return res;
	}
};


int main() {
	XERUS_LOG(simpleALS,"Begin Building Operator ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");

	size_t dim = 26; // 16 electron, 8 electron pairs
	size_t nob = 13;
	XERUS_LOG(simpleMALS,"Load Data ...");

	Data_CH2 test(dim);
	XERUS_LOG(info,test.P_L[0][1].get_component(0));
	XERUS_LOG(info,"V(4,1,3,2) = " << test.V[{6,0,4,2}]);
	XERUS_LOG(info,"V(4,1,3,2) = " << test.V[{6,0,2,4}]);
	XERUS_LOG(info,"w(4,1,3,2) = " << test.w(6,0,4,2));
	XERUS_LOG(info,"w(4,2,3,1) = " << test.w(6,2,4,0));
	XERUS_LOG(info,"w(1,2,3,4) = " << test.w(3,0,2,1) - test.w(3,1,2,0));
	XERUS_LOG(info,"2*w(1,2,3,4) = " << 2* test.w(1,2,3,5) );



  return 0;

}

