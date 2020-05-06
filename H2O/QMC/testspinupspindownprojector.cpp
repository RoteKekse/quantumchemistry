#include <xerus.h>
#include <chrono>


#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"
#include "../../classes/GradientMethods/ALSres.cpp"




int main(){
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
	read_from_disc("../data/hf_gradient_48.tttensor",start2);
	read_from_disc("../data/hf_gradient_48.tttensor",start3);
	read_from_disc("../data/hf_gradient_48.tttensor",start4);
	start/= start.frob_norm();
	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));
	XERUS_LOG(info,start.ranks());


	XERUS_LOG(info,"Round start vector to " << eps << " keeping sing values bigger than " << eps/std::sqrt(d-1));
	start/= start.frob_norm();
	for (value_t ee = 0.01; ee <= eps ; ee+=0.01){
		start.round(ee);
		//start/= start.frob_norm();
	}
	start/= start.frob_norm();

	start2/= start2.frob_norm();
	for (value_t ee = 0.01; ee <= eps ; ee+=0.03){
		start2.round(ee);
		//start2/= start2.frob_norm();
	}
	start2.round(eps);
	start2/= start2.frob_norm();

	start3/= start3.frob_norm();
	start3.round(eps);
	start3/= start3.frob_norm();


	XERUS_LOG(info,"Rounding with the help of ALS");
	TTTensor start4_rounded = start3;
	for (size_t i = 0; i < roundIter;++i){
		getRes(id,start4,id,0.0,start4_rounded);
		XERUS_LOG(info, "Run " << i);
		XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start4_rounded,start4_rounded)/contract_TT(id,start4_rounded,start4_rounded));
		XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start4_rounded,start4_rounded)/contract_TT(id,start4_rounded,start4_rounded));
		XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start4_rounded,start4_rounded)/contract_TT(id,start4_rounded,start4_rounded));
		XERUS_LOG(info, "Error = " << (start4-start4_rounded).frob_norm());
	}

	XERUS_LOG(info,"End Rounding with the help of ALS");


	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start,start));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start,start));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start,start));
	XERUS_LOG(info,start.ranks());

	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start2,start2));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start2,start2));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start2,start2));
	XERUS_LOG(info,start2.ranks());

	XERUS_LOG(info,"Particle number start       " << std::setprecision(16) << contract_TT(P,start3,start3));
	XERUS_LOG(info,"Particle number up start    " << std::setprecision(16) << contract_TT(Pup,start3,start3));
	XERUS_LOG(info,"Particle number down start  " << std::setprecision(16) << contract_TT(Pdown,start3,start3));
	XERUS_LOG(info,start3.ranks());

	phi -= alpha*start;
	phi/=phi.frob_norm();
	phi.move_core(0);
	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));
	XERUS_LOG(info,phi.ranks());


	TangentialOperation top(phi);
	auto tang1 = top.localProduct(start,id);
	auto tang2 = top.localProduct(start,Hs);
	auto tang1TT = top.builtTTTensor(tang1);
	auto tang2TT = top.builtTTTensor(tang2);
	TTTensor tang3TT;
	Index i1,j1;
	tang3TT(i1&0) = Hs(i1/2,j1/2)*start(j1&0);
	XERUS_LOG(info,tang3TT.ranks());

	auto tang4 = top.localProduct(tang3TT,id);
	auto tang4TT = top.builtTTTensor(tang4);

	XERUS_LOG(info, "Norm start projected    " << tang1TT.frob_norm());
	XERUS_LOG(info, "Norm Hsstart projected  " << tang2TT.frob_norm());
	XERUS_LOG(info, "Norm Hsstart            " << tang3TT.frob_norm());
	XERUS_LOG(info, "Norm Hsstart projected2 " << tang4TT.frob_norm());
	tang1TT/= tang1TT.frob_norm();
	tang2TT/= tang2TT.frob_norm();
	tang3TT/= tang3TT.frob_norm();
	tang4TT/= tang4TT.frob_norm();

	XERUS_LOG(info,"Particle number start projected       " << std::setprecision(16) << contract_TT(P,tang1TT,tang1TT));
	XERUS_LOG(info,"Particle number up start projected    " << std::setprecision(16) << contract_TT(Pup,tang1TT,tang1TT));
	XERUS_LOG(info,"Particle number down start projected  " << std::setprecision(16) << contract_TT(Pdown,tang1TT,tang1TT));

	XERUS_LOG(info,"Particle number Hs start projected       " << std::setprecision(16) << contract_TT(P,tang2TT,tang2TT));
	XERUS_LOG(info,"Particle number up Hs start projected    " << std::setprecision(16) << contract_TT(Pup,tang2TT,tang2TT));
	XERUS_LOG(info,"Particle number down Hs start projected  " << std::setprecision(16) << contract_TT(Pdown,tang2TT,tang2TT));

	XERUS_LOG(info,"Particle number Hs start        " << std::setprecision(16) << contract_TT(P,tang3TT,tang3TT));
	XERUS_LOG(info,"Particle number up Hs start     " << std::setprecision(16) << contract_TT(Pup,tang3TT,tang3TT));
	XERUS_LOG(info,"Particle number down Hs start   " << std::setprecision(16) << contract_TT(Pdown,tang3TT,tang3TT));

	XERUS_LOG(info,"Particle number Hs start        " << std::setprecision(16) << contract_TT(P,tang4TT,tang4TT));
	XERUS_LOG(info,"Particle number up Hs start     " << std::setprecision(16) << contract_TT(Pup,tang4TT,tang4TT));
	XERUS_LOG(info,"Particle number down Hs start   " << std::setprecision(16) << contract_TT(Pdown,tang4TT,tang4TT));

	return 0;
}
