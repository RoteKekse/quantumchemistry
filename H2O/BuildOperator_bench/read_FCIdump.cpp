#include <xerus.h>

#include <queue>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace xerus;
using xerus::misc::operator<<;
/*
 * Loads the 1 electron integral
 */
xerus::Tensor load_1e_int(size_t d, std::string path){
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

/*
* Loads the 2 electron integral
*/
xerus::Tensor load_2e_int(size_t d, std::string path){
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


int main(){

	size_t d = 48;
	std::string path = "../FCIDUMP.h2o_24";
	Tensor T  = load_1e_int(d,path),V=load_2e_int(d,path);





	std::string name = "../data/T_H2O_48_bench.tensor";
	std::ofstream write(name.c_str());
	misc::stream_writer(write,T,xerus::misc::FileFormat::BINARY);
	write.close();

	name = "../data/V_H2O_48_bench.tensor";
	std::ofstream write2(name.c_str());
	misc::stream_writer(write2,V,xerus::misc::FileFormat::BINARY);
	write2.close();

	return 0;
}

