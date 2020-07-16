#include <xerus.h>
#include <chrono>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"
#include "../../classes/GradientMethods/ALSres.cpp"




int main(){
	srand (time(NULL));
	size_t d = 48,p = 8,iterations = 1e6,iterations2 = 100*iterations,roundIter = 10, rank = 10;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=1.0,ev_app_tmp;
	std::vector<size_t> hf_sample = {0,1,2,3,22,23,30,31};
	value_t alpha = 0.1,beta;

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(d)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(2*d,2));
	auto P = particleNumberOperator(d);
	auto Pup = particleNumberOperatorUp(d);
	auto Pdown = particleNumberOperatorDown(d);

	xerus::TTTensor phi,res,res_last,start,start2,start3,start4;
	phi = makeUnitVector(hf_sample,d);
	read_from_disc("../data/hf_gradient_48.tttensor",start);

	start/= start.frob_norm();
	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));
	XERUS_LOG(info,start.ranks());

	size_t idx = 2;
	start.move_core(idx+1);
	auto split = start.chop(idx+1);


	TTTensor start_first(std::vector<size_t>(idx,2));
	for (size_t i = 0; i < idx-1; ++i){
		Tensor tensor(*split.first.nodes[i+1].tensorObject);
		XERUS_LOG(info,tensor.dimensions);
		start_first.set_component(i,tensor);
	}
	Tensor tensor(*split.first.nodes[idx].tensorObject);
	XERUS_LOG(info,tensor.dimensions);
	auto Psplit = particleNumberOperator(idx);
	auto Psplitup = particleNumberOperatorUp(idx);
	auto Psplitdown = particleNumberOperatorDown(idx);
	auto idsplit = xerus::TTOperator::identity(std::vector<size_t>(2*idx,2));
	for (size_t slice = 0; slice < tensor.dimensions[2]; slice++){
		XERUS_LOG(info,slice);

		Tensor tmp = tensor;
		tmp.fix_mode(2,slice);
		tmp.reinterpret_dimensions({tmp.dimensions[0],tmp.dimensions[1],1});
		start_first.set_component(idx-1,tmp);
		tmp = Tensor(start_first);
		tmp.reinterpret_dimensions({(size_t) std::pow(2,idx)});
		XERUS_LOG(info,"\n" << tmp);
		start_first/= start_first.frob_norm();
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplit,start_first,start_first));
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplitup,start_first,start_first));
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplitdown,start_first,start_first));
	}

	start.move_core(idx-1);
	split = start.chop(idx-1);


	TTTensor start_last(std::vector<size_t>(d-idx,2));
	for (size_t i = 1; i < d-idx; ++i){
		Tensor tensor(*split.second.nodes[i].tensorObject);
		XERUS_LOG(info,tensor.dimensions);
		start_last.set_component(i,tensor);
	}
	tensor = Tensor(*split.second.nodes[0].tensorObject);
	XERUS_LOG(info,tensor.dimensions);
	auto Psplit2 = particleNumberOperator(d-idx);
	auto Psplitup2 = particleNumberOperatorUp(d-idx);
	auto Psplitdown2 = particleNumberOperatorDown(d-idx);
	for (size_t slice = 0; slice < tensor.dimensions[0]; slice++){
		XERUS_LOG(info,slice);

		Tensor tmp = tensor;
		tmp.fix_mode(0,slice);
		tmp.reinterpret_dimensions({1,tmp.dimensions[0],tmp.dimensions[1]});
		start_last.set_component(0,tmp);

		start_last/= start_last.frob_norm();
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplit2,start_last,start_last));
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplitup2,start_last,start_last));
		XERUS_LOG(info,"Particle number split       " << std::setprecision(16) << contract_TT(Psplitdown2,start_last,start_last));
	}



	start.move_core(0);
	for (size_t i = 0; i < d;++i){
		Tensor ten = start.get_component(i);
		for (size_t j = 0; j< ten.dimensions[0]*ten.dimensions[1]*ten.dimensions[2];++j)
			if (std::abs(ten[j]) < 10e-10)
				ten[j] = 0;
		start.set_component(i,ten);
	}
//	Tensor test(start.get_component(idx).dimensions);
//	test[{0,0,1}] = (double)rand() / RAND_MAX;
//	test[{0,0,2}] = (double)rand() / RAND_MAX;
//	test[{0,1,0}] = (double)rand() / RAND_MAX;
////	test[{1,0,4}] = (double)rand() / RAND_MAX;
//	//test[{1,0,5}] = (double)rand() / RAND_MAX;
//	//test[{1,0,6}] = (double)rand() / RAND_MAX;
////	test[{1,1,1}] = (double)rand() / RAND_MAX;
////	test[{1,1,2}] = (double)rand() / RAND_MAX;
////	test[{1,1,3}] = (double)rand() / RAND_MAX;
////	test[{2,0,4}] = (double)rand() / RAND_MAX;
////	test[{2,0,5}] = (double)rand() / RAND_MAX;
////	test[{2,0,6}] = (double)rand() / RAND_MAX;
////	test[{2,1,1}] = (double)rand() / RAND_MAX;
////	test[{2,1,2}] = (double)rand() / RAND_MAX;
////	test[{2,1,3}] = (double)rand() / RAND_MAX;
//	test[{3,1,4}] = (double)rand() / RAND_MAX;
////	test[{3,1,5}] = (double)rand() / RAND_MAX;
////	test[{3,1,6}] = (double)rand() / RAND_MAX;
//
//	//test[{3,0,6}] = 1;
//
//	start.set_component(idx,test);
	Tensor t = start.get_component(idx);
	XERUS_LOG(info,"\n" << t);
	for (size_t i = 0; i < t.dimensions[0];++i){
		for (size_t j = 0; j < t.dimensions[1];++j){
			for (size_t k = 0; k < t.dimensions[2];++k){
				if (std::abs(t[{i,j,k}]) > 10e-10)
					XERUS_LOG(info,i <<" " <<j<< " "<< k<<": " << t[{i,j,k}]);


			}
		}
	}
	XERUS_LOG(info, t.dimensions);

	t.reinterpret_dimensions({t.dimensions[1],t.dimensions[0],t.dimensions[2]});
	t.reinterpret_dimensions({t.dimensions[0]*t.dimensions[1],t.dimensions[2]});
	XERUS_LOG(info,"\n" << t);
	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start)/contract_TT(id,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start)/contract_TT(id,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start)/contract_TT(id,start,start));
	return 0;
}
