#include <xerus.h>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>


using namespace xerus;
using xerus::misc::operator<<;


xerus::Tensor make_H_CH2(size_t nob);
xerus::Tensor make_V_CH2(size_t nob);
xerus::Tensor make_Nuc_CH2();



xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);


xerus::TTOperator build_partial_operator(size_t i, size_t j, size_t d, value_t estimate, const Tensor &H, const Tensor &V);

Tensor get_left(const TTTensor &a, const TTTensor &b, const size_t _position) {
	Index i1, i2, i3, j1 , j2, j3, k1, k2;
	auto tmp = Tensor::ones({1,1});
	for ( size_t i = 0; i < _position; ++i){
		const Tensor &ai = a.get_component(i);
		const Tensor &bi = b.get_component(i);
		tmp(i1,i2) = tmp(j1,j2) * ai(j1,k1,i1) * bi(j2,k1,i2);
	}
	return tmp;
}


Tensor get_right(const TTTensor &a, const TTTensor &b, const size_t _position,size_t d) {
	Index i1, i2, i3, j1 , j2, j3, k1, k2;
	auto tmp = Tensor::ones({1,1});
	for ( size_t i = d - 1; i > _position; --i){
		const Tensor &ai = a.get_component(i);
		const Tensor &bi = b.get_component(i);
		tmp(i1,i2) =  ai(i1,k1,j1) * bi(i2,k1,j2)  * tmp(j1,j2);
	}
	return tmp;
}


/*
 * Main!!
 */
int main() {
	XERUS_LOG(simpleALS,"Begin Tests for ordering ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");


	size_t d = 26; // 16 electron, 8 electron pairs
	size_t nob = 13;
	double eps = 10e-6;
	size_t start_rank = 4;
	size_t max_rank = 16;
	size_t wsize = 5;
	value_t est = 12.0;
	value_t rate = 10e-13;
	size_t max_iter = 1000;

	XERUS_LOG(simpleMALS,"Load Matrix ...");
	xerus::Tensor H = make_H_CH2(nob);
  xerus::Tensor V = make_V_CH2(nob);
  xerus::Tensor N = make_Nuc_CH2();

	xerus::TTOperator op_full;
	std::string name = "../hamiltonian_CH2_" + std::to_string(d) +"_full_optimized.ttoperator";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,op_full,xerus::misc::FileFormat::BINARY);

  XERUS_LOG(info,"Nuc Pot = " << N[0]);
	xerus::Index i1,i2,i3,ii,jj, kk, ll;
  xerus::TTTensor phi = xerus::TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d - 1,start_rank));
	xerus::Tensor ritz;
	xerus::TTTensor res;
	for (size_t itr = 0; itr < max_iter; ++itr){
	  phi /= phi.frob_norm();
		size_t i = std::rand() % d;
		size_t k;
		if(std::rand() % 10 == 0)
			k = i;
		else
			k = std::rand() % d;
		xerus::TTOperator op_tmp =  build_partial_operator(i,k,d,est,H,V);
		res(ii&0) = op_tmp(ii/2,jj/2) * phi(jj&0);
		ritz() = phi(ii&0)*res(ii&0);
		res = rate*(res - ritz[0]*phi);

		for (size_t pos = 0; pos < d; ++pos){
			phi.move_core(pos);
		  phi /= phi.frob_norm();
		  auto resl = get_left(phi,res,pos);
		  auto resr = get_right(phi,res,pos,d);
		  auto resm = res.get_component(pos);
		  xerus::Tensor resc;
		  resc(i1,i2,i3) =resl(i1,ii)*resm(ii,i2,jj)*resr(i3,jj);
		  phi.component(pos) += resc;
		}

		XERUS_LOG(info,"EV = " << std::setprecision(16) << frob_norm(phi(ii&0)*op_full(ii/2,jj/2) * phi(jj&0))-est);
	}

	//XERUS_LOG(info, "The ranks of op are " << op.ranks() );
	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );

	return 0;
}

/*
 *
 *
 *
 * Functions
 *
 *
 */


xerus::TTOperator build_partial_operator(size_t i, size_t k, size_t d, value_t estimate, const Tensor &H, const Tensor &V){
	auto op = xerus::TTOperator(std::vector<size_t>(2*d,2));
	value_t val = H[{i , k}];
	op += val * return_one_e_ac(i,k,d);
	for (size_t j = 0; j < d; j++){
			for (size_t l = 0; l < d; l++){
				value_t val = V[{i,j,k,l}];
				if (std::abs(val) < 10e-14 )//|| a == b || c: == d) // TODO check i == j || k == l
					continue;
				op += 0.5*val * return_two_e_ac(i,j,l,k,d);
		}
	}
	op += estimate*xerus::TTOperator::identity(std::vector<size_t>(2*d,2));
	op.round(10e-14);
	return op;
}



xerus::Tensor make_Nuc_CH2(){
	auto Nuc = xerus::Tensor({1});
	std::string line;
	std::ifstream input;
	input.open ("../FCIDUMP.ch2_13");
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

xerus::Tensor make_H_CH2(size_t nob){
	auto H = xerus::Tensor({2*nob,2*nob});
	auto H_tmp = xerus::Tensor({nob,nob});
	std::string line;
	std::ifstream input;
	input.open ("../FCIDUMP.ch2_13");
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


xerus::Tensor make_V_CH2(size_t nob){
	auto V = xerus::Tensor({2*nob,2*nob,2*nob,2*nob});
	auto V_tmp = xerus::Tensor({nob,nob,nob,nob});
	auto V_tmp2 = xerus::Tensor({nob,nob,nob,nob});
	std::string line;
	std::ifstream input;
	input.open ("../FCIDUMP.ch2_13");
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

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d){ // TODO write tests for this
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


