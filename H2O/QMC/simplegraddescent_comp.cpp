#include <xerus.h>

#include "../../classes/QMC/tangential_parallel.cpp"
#include "../../classes/QMC/basic.cpp"
#include "../../classes/QMC/tangentialOperation.cpp"

#include "../../classes/loading_tensors.cpp"
#include "../../classes/helpers.cpp"

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);


int main(){
	size_t nob = 24,num_elec = 8,iterations = 1e4, numIter = 20,rank=20;
	value_t ev, shift = 25.0, ev_app, ev_ex, eps=1,ev_app_tmp;
	std::string path_T = "../data/T_H2O_48_bench_single.tensor";
	std::string path_V= "../data/V_H2O_48_bench_single.tensor";
	std::vector<size_t> sample = { 0, 1,2,3,22,23,30,31 };
	value_t nuc = -52.4190597253;
	value_t alpha = 0.1,beta,rHx,rHr,rx,rr;

	XERUS_LOG(info,"Loading Start vector from disc");
	TTTensor start, phi,res,step;
	read_from_disc("../data/hf_gradient_48.tttensor",start);
	phi = makeUnitVector(sample,2*nob);


	auto P = particleNumberOperator(2*nob);
	auto Pup = particleNumberOperatorUp(2*nob);
	auto Pdown = particleNumberOperatorDown(2*nob);
	auto id = xerus::TTOperator::identity(std::vector<size_t>(4*nob,2));

	xerus::TTOperator Hs;
	std::string name2 = "../data/hamiltonian_H2O_" + std::to_string(2*nob)  +"_full_shifted_benchmark.ttoperator";
	read_from_disc(name2,Hs);
	XERUS_LOG(info, "Eigenvalue start   " << std::setprecision(8) << contract_TT(Hs,phi,phi)- shift +nuc);

	XERUS_LOG(info,"Round start vector to " << eps << " keeping sing values bigger than " << eps/std::sqrt(2*nob-1));
	start/= start.frob_norm();
	for (value_t ee = 0.05; ee <= eps ; ee+=0.05){
		start.round(ee);
		start/= start.frob_norm();
	}
	start/= start.frob_norm();
	phi -= alpha*start;
	phi/=phi.frob_norm();
	phi.move_core(0);
	XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
	XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
	XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));
	XERUS_LOG(info,phi.ranks());

	Tangential tang(2*nob,num_elec,iterations,path_T,path_V,shift,sample,phi);
	TangentialOperation top(phi);
	tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
	tang.uvP.xbase.second = top.xbasis[1];
	std::vector<Tensor> tang_app, tang_app_old;
	ev_app = contract_TT(Hs,phi,phi);
	for (size_t i = 0; i < numIter;++i){
		XERUS_LOG(info, "Eigenvalue exact   " << std::setprecision(8) << ev_app- shift +nuc);
		tang_app = tang.get_tangential_components(ev_app,0.0001);

		if (i == 0){
			res = top.builtTTTensor(tang_app);
		} else {
			tang_app_old = top.localProduct(res,id);
			beta = frob_norm(tang_app)/frob_norm(tang_app_old); //Fletcher Reeves update
			add(tang_app,tang_app_old, beta);
			res = top.builtTTTensor(tang_app);
		}

		step = res/res.frob_norm();
		ev_app_tmp = ev_app;

		rHx = contract_TT(Hs,step,phi);
		rHr = contract_TT(Hs,step,step);
		rx = contract_TT(id,step,phi);
		alpha = get_stepsize(ev_app,rHr,rHx,1.0,1.0,rx);

		XERUS_LOG(info, "alpha = " << alpha);
		phi = phi -  alpha*step;

		phi /= phi.frob_norm();
		phi.round(rank);
		phi /= phi.frob_norm();
		tang.update(phi);
		ev_app = contract_TT(Hs,phi,phi);
		XERUS_LOG(info, "Eigenvalue exact tmp   " << std::setprecision(8) << ev_app- shift +nuc);
		XERUS_LOG(info,"Particle number phi updated       " << std::setprecision(16) << contract_TT(P,phi,phi));
		XERUS_LOG(info,"Particle number up phi updated    " << std::setprecision(16) << contract_TT(Pup,phi,phi));
		XERUS_LOG(info,"Particle number down phi updated  " << std::setprecision(16) << contract_TT(Pdown,phi,phi));


		XERUS_LOG(info,"ranks = \n" <<phi.ranks());


		//update
		top.update(phi);
		tang.update(phi);
		tang.uvP.xbase.first = top.xbasis[0]; //use same orthogonalization!!!
		tang.uvP.xbase.second = top.xbasis[1];
		if (alpha < 0.1)
			alpha *= 2;
	}


	ev_app = contract_TT(Hs,phi,phi);
	XERUS_LOG(info, "Eigenvalue exact   " << std::setprecision(8) << ev_app- shift +nuc);








	return 0;
}

double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx){
	double a = rFr*rHx-rHr*rFx;
	double b = rHr*xFx-rFr*xHx;
	double c = rFx*xHx-rHx*xFx;

	double disc = b*b-4*a*c;
	double alpha1 = (-b + std::sqrt(disc))/(2*a);
	double alpha2 = (-b - std::sqrt(disc))/(2*a);
	return alpha1;
}

