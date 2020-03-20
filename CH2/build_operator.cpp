#include <xerus.h>
#include <Eigen/Dense>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

#include <mpi.h>

using namespace xerus;
using xerus::misc::operator<<;

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);

xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);

xerus::Tensor make_H_CH2(size_t nob);
xerus::Tensor make_V_CH2(size_t nob);
xerus::Tensor make_Nuc_CH2();

xerus::Tensor make_Permutation(std::vector<size_t> _perm, size_t dim);


int main() {
	XERUS_LOG(simpleALS,"Begin Building Operator ...");
	XERUS_LOG(simpleALS,"---------------------------------------------------------------");



	size_t dim = 26; // 16 electron, 8 electron pairs
	size_t nob = 13;
	double eps = 10e-6;
	size_t start_rank = 2;
	size_t max_rank = 79;
	size_t wsize = 4;
	XERUS_LOG(simpleMALS,"Load Data ...");
	xerus::Tensor H = make_H_CH2(nob);
  xerus::Tensor V = make_V_CH2(nob);
  xerus::Tensor N = make_Nuc_CH2();
  XERUS_LOG(info,"Nuc Pot = " << N[0]);

  xerus::Index i1,i2,i3,i4, ii,jj,kk,ll;
	xerus::Tensor perm_T = make_Permutation({0, 1, 5, 4, 3, 2, 6, 7, 9, 8, 12, 13, 10, 11, 14, 15, 17, 16, 19, 18, 20, 21, 22, 23, 25, 24}, dim); //log best
	H(i1,i2) = perm_T(i1,jj) * H(jj,kk) *perm_T(i2,kk);
	V(i1,i2,i3,i4) = perm_T(i1,ii) * perm_T(i2,jj) * perm_T(i3,kk) * perm_T(i4,ll) * V(ii,jj,kk,ll);

	auto opH = xerus::TTOperator(std::vector<size_t>(2*dim,2));

  	MPI_Init(NULL, NULL);
  	int world_rank;
  	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  	int world_size;
  	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  	if (world_rank == 0){
			for (size_t i = 0; i < dim; i++){
					for (size_t j = 0; j < dim; j++){
							value_t val = H[{i , j}];
							opH += val * return_one_e_ac(i,j,dim);
					}
			}
  	}
  	int len = static_cast<int>(dim) / world_size;
  	int end = world_rank == world_size - 1 ? dim : world_rank * len + len;
  	auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2));
  	for (size_t i = world_rank * len; i < end; i++){
  		for (size_t j = 0; j < dim; j++){
  			XERUS_LOG(info,  i << " " << j);
  			for (size_t k = 0; k < dim; k++){
  				for (size_t l = 0; l < dim; l++){
  					value_t val = V[{i,j,k,l}];
  					if (std::abs(val) < eps )//|| a == b || c: == d) // TODO check i == j || k == l
  						continue;
  					opV += 0.5*val * return_two_e_ac(i,j,l,k,dim);
  				}
  			}
  		}
  	}
  	std::string name2 = "hamiltonian_CH2_" + std::to_string(dim) +"_" + std::to_string(world_rank) +".ttoperator";
  	std::ofstream write(name2.c_str() );
  	xerus::misc::stream_writer(write,opV,xerus::misc::FileFormat::BINARY);
  	write.close();
  	MPI_Barrier(MPI_COMM_WORLD);

  	if(world_rank == 0){
			auto opV = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
			for(size_t i = 0; i < wsize; ++i){
				XERUS_LOG(simpleMALS, "---- Loading partial operator " << i << " ----");
				auto tmp = xerus::TTOperator(std::vector<size_t>(2*dim,2)); // TT operator initialized with 0
				std::string name = "hamiltonian_CH2_"+std::to_string(dim)+"_" +std::to_string(i)+ ".ttoperator";
				std::ifstream read(name.c_str());
				misc::stream_reader(read,tmp,xerus::misc::FileFormat::BINARY);
				read.close();
				opV += tmp;
			}
			xerus::TTOperator op = opH + opV;

			std::string name2 = "hamiltonian_CH2_" + std::to_string(dim) +"_full_optimized.ttoperator";
			std::ofstream write(name2.c_str() );
			xerus::misc::stream_writer(write,op,xerus::misc::FileFormat::BINARY);
			write.close();
  	}


    MPI_Finalize();

    return 0;

}

xerus::Tensor make_Permutation(std::vector<size_t> _perm, size_t dim){
	auto perm = xerus::Tensor({dim,dim});
	for (size_t i = 0; i < dim; ++i)
		perm[{i,_perm[i]}] = 1;
	return perm;
}


xerus::Tensor make_Nuc_CH2(){
	auto Nuc = xerus::Tensor({1});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.ch2_13");
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
	input.open ("FCIDUMP.ch2_13");
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
	input.open ("FCIDUMP.ch2_13");
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

