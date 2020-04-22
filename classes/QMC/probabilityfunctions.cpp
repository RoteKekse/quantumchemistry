#include <xerus.h>
#include <stdlib.h>
#include "contractpsihek.cpp"
#include "unitvectorprojection.cpp"
#include "../../classes/containerhash.cpp"
#include <unordered_map>


using namespace xerus;
using xerus::misc::operator<<;

#pragma once




class ProbabilityFunction {
	public:
		std::unordered_map<std::vector<size_t>,value_t,container_hash<std::vector<size_t>>> values;
	protected:
		ProbabilityFunction(){}
		virtual value_t P(std::vector<size_t> sample) = 0;
		virtual ~ProbabilityFunction(){}

};



class DummyProbabilityFunction : ProbabilityFunction{
	public:
		DummyProbabilityFunction(){}
		value_t P(std::vector<size_t> sample) override {
			return 0.5;
		}
};

class ProjectorProbabilityFunctionBase : public ProbabilityFunction{
	public:
		std::pair<TTTensor, TTTensor> xbase;
		size_t d;
		size_t position;
		bool proj;
	private:
		Tensor leftxStack,rightxStack;
		Index i1,i2,i3,A1,A2,A3,A4,k1,k2,j1,j2,j3,j4,j5,j6;
	public:
		ProjectorProbabilityFunctionBase(TTTensor _psi, size_t _pos, bool _proj) : d(_psi.order()), position(_pos), proj(_proj){
			_psi.move_core(0,true);
			xbase.first = _psi;
			_psi.move_core(d-1,true);
			xbase.second = _psi;
		}

		void update(TTTensor psi, size_t _position){
			psi.move_core(0,true);
			xbase.first = psi;
			psi.move_core(d-1,true);
			xbase.second = psi;
			position = _position;
		}


		Tensor localProduct(std::vector<size_t> sample) {
			Tensor comp,tmp,unit,xi;
			leftxStack = Tensor::ones({1});
			rightxStack = Tensor::ones({1});
#pragma omp parallel sections
		{
#pragma omp section
			{
				for (size_t pos=d-1; pos>position;pos--){
					tmp = xbase.first.get_component(pos);
					if (std::find(sample.begin(), sample.end(), pos) != sample.end())
						tmp.fix_mode(1,1);
					else
						tmp.fix_mode(1,0);
					rightxStack(i1) = tmp(i1,j1)*rightxStack(j1);
				}
			}
#pragma omp section
			{
				for (size_t pos=0; pos<position;pos++){
					tmp = xbase.second.get_component(pos);
					if (std::find(sample.begin(), sample.end(), pos) != sample.end())
						tmp.fix_mode(1,1);
					else
						tmp.fix_mode(1,0);
					leftxStack(i1) = tmp(j1,i1)*leftxStack(j1);
				}
			}
#pragma omp section
			{
				if (std::find(sample.begin(), sample.end(), position) != sample.end())
					unit = Tensor::dirac({2},1);
				else
					unit = Tensor::dirac({2},0);
			}
		}

			comp(i1,i2,i3) = leftxStack(i1) * unit(i2)  * rightxStack(i3);
			// projection
			if (position < d - 1 and proj){
				xi=xbase.second.get_component(position);
				tmp(i1,i2,i3)= xi(i1,i2,k1)*xi(j1,j2,k1)*comp(j1,j2,i3);
				comp -= tmp;
			}
			return comp;
		}



};

class ProjectorProbabilityFunction : public ProjectorProbabilityFunctionBase{
	public:
		ProjectorProbabilityFunction(TTTensor _psi, size_t _pos, bool _proj) :
					ProjectorProbabilityFunctionBase(_psi, _pos, _proj){}

		value_t P(std::vector<size_t> sample) override {
			auto itr = values.find(sample);
			if (itr == values.end()){
				values[sample] =  localProduct(sample).frob_norm();
			}
			return std::pow(values[sample],2);
		}
};

class ProjectorProbabilityFunction2 : public ProjectorProbabilityFunctionBase{
	public:
		ContractPsiHek builder;

	public:
		ProjectorProbabilityFunction2(TTTensor _psi, size_t _pos, bool _proj, ContractPsiHek& _builder) :
			ProjectorProbabilityFunctionBase(_psi, _pos, _proj), builder(_builder){}

		value_t P(std::vector<size_t> sample) override {
			auto itr = values.find(sample);
			if (itr == values.end()){
				builder.reset(sample);
				value_t dk = builder.diagionalEntry();
				values[sample] =  localProduct(sample).frob_norm()/dk;
			}

			return std::pow(values[sample],2);
		}


};

class ProjectorProbabilityFunction3 : public ProjectorProbabilityFunctionBase{
	public:
		TTOperator Fock_inv;

	public:
		ProjectorProbabilityFunction3(TTTensor _psi, size_t _pos, bool _proj, TTOperator& _Fock_inv) :
			ProjectorProbabilityFunctionBase(_psi, _pos, _proj), Fock_inv(_Fock_inv){}

		value_t P(std::vector<size_t> sample) override {
			auto itr = values.find(sample);
			if (itr == values.end()){
				auto idx = makeIndex(sample);
				idx.resize(2 * d);
				std::copy_n(idx.begin(), d, idx.begin() + d);
				value_t dk = Fock_inv[idx];
				values[sample] =  localProduct(sample).frob_norm()*dk;
			}

			return std::pow(values[sample],2);
		}

		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}

};

class ProjectorProbabilityFunction4 : public ProjectorProbabilityFunctionBase{
	public:
		TTOperator Fock;

	public:
		ProjectorProbabilityFunction4(TTTensor _psi, size_t _pos, bool _proj, TTOperator& _Fock) :
			ProjectorProbabilityFunctionBase(_psi, _pos, _proj), Fock(_Fock){}



		value_t P(std::vector<size_t> sample) override {
			auto itr = values.find(sample);
			if (itr == values.end()){
				auto idx = makeIndex(sample);
				idx.resize(2 * d);
				std::copy_n(idx.begin(), d, idx.begin() + d);
				value_t dk = Fock[idx];
				values[sample] =  localProduct(sample).frob_norm()/dk;
			}

			return std::pow(values[sample],2);
		}

		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}

};



class PsiProbabilityFunction : public ProbabilityFunction{
	public:
		TTTensor psi;
		size_t d;
		size_t p_up;
		size_t p_down;
		size_t count;

	public:
		PsiProbabilityFunction(TTTensor _psi): d(_psi.order()), p_up(0), p_down(0), count(0){
			psi = _psi;
		}

		value_t P(std::vector<size_t> sample) override {
			auto itr = values.find(sample);
			if (itr == values.end()){
				std::vector<size_t> idx = makeIndex(sample);
				values[sample] =  psi[idx];
			}
			return std::pow(values[sample],2);
		}

		value_t Pval(std::vector<size_t> sample) {
			std::vector<size_t> idx = makeIndex(sample);
			return psi[idx];
		}

		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}


};


class PsiProbabilityFunction2 : public ProbabilityFunction{
	public:
		TTTensor psi;
		size_t d;
		size_t p_up;
		size_t p_down;
		size_t count;

	public:
		PsiProbabilityFunction2(TTTensor _psi): d(_psi.order()), p_up(0), p_down(0), count(0){
			psi = _psi;
		}



		value_t P(std::vector<size_t> sample) override {
			auto index = makeIndex(sample);
			auto itr = values.find(index);
			if (itr == values.end()){
				XERUS_LOG(info,sample);
				preparePsiEval(index);
			}

			return std::pow(values[index],2);
		}

		value_t Pval(std::vector<size_t> sample) {
			std::vector<size_t> idx = makeIndex(sample);
			return psi[idx];
		}

		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			p_up = 0;
			p_down = 0;
			for (size_t i : sample){
				if (i < d)
					index[i] = 1;
				if (i%2 == 0)
					p_up++;
				else
					p_down++;
			}
			return index;
		}


		void preparePsiEval(std::vector<size_t> idx){ 			// TODO can one keep the lower contractions for different e_ks??
			Index r1,r2,r3;
			count++;
			XERUS_LOG(info,count);


			std::queue<std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>> queue;
			std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>> data_tmpl;
			for (size_t i = 0; i < 3*3*(p_up+1)*(p_down+1); ++i){
				std::vector<std::pair<std::vector<size_t>,Tensor>> tmp;
				data_tmpl.emplace_back(tmp);
			}

			// initialize queue with slices of TT Tensor
			for (size_t i = 0; i < d; ++i){
				auto psi0 = psi.get_component(i);
				auto psi1 = psi.get_component(i);
				psi0.fix_mode(1,0);
				psi1.fix_mode(1,1);
				auto data = data_tmpl;
				data[getIndex(idx[i] == 1 ? 1 : 0,0,0,0)].emplace_back(std::pair<std::vector<size_t>,Tensor>({0},psi0));
				data[getIndex(0,idx[i] == 1 ? 0 : 1,(i+1)%2,i%2)].emplace_back(std::pair<std::vector<size_t>,Tensor>({1},psi1));
				queue.push(std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>(i,data));
			}

			bool finished = false;
			size_t count = 0;
			while (not finished){
				finished = queue.size() == 2 ? true : false;
				auto elm1 = queue.front();
				queue.pop();
				auto elm2 = queue.front();
				queue.pop();
				if (elm1.first > elm2.first){
					queue.push(elm1);
					elm1 = elm2;
					elm2 = queue.front();
					queue.pop();
				}

				size_t pos = elm1.first;

				auto data = data_tmpl;
				for (size_t i1 = 0; i1 < 3; ++i1){
					for (size_t j1 = 0; j1 < 3; ++j1){
						for (size_t k1 = 0; k1 <= p_up; ++k1){
							for (size_t l1 = 0; l1 <= p_down; ++l1){
								for (auto const& tuple1 : elm1.second[getIndex(i1,j1,k1,l1)]){
									for (size_t i2 = 0; i2 < 3-i1; ++i2){
										for (size_t j2 = 0; j2 < 3-j1; ++j2){
											if (not finished){
												for (size_t k2 = 0; k2 <= p_up-k1; ++k2){
													for (size_t l2 = 0; l2 <= p_down-l1; ++l2){
														for (auto const& tuple2 : elm2.second[getIndex(i2,j2,k2,l2)]){
															std::vector<size_t> idx_new(tuple1.first);
															idx_new.insert(idx_new.end(),tuple2.first.begin(),tuple2.first.end());
															Tensor tmp;
															tmp(r1,r3) = tuple1.second(r1,r2)*tuple2.second(r2,r3);
															data[getIndex(i1+i2,j1+j2,k1+k2,l1+l2)].emplace_back(std::pair<std::vector<size_t>,Tensor>(idx_new,std::move(tmp)));
											}}}}
											else {
												for (auto const& tuple2 : elm2.second[getIndex(i2,j2,p_up-k1,p_down-l1)]){
													std::vector<size_t> idx_new(tuple1.first);
													idx_new.insert(idx_new.end(),tuple2.first.begin(),tuple2.first.end());
													Tensor tmp;
													XERUS_LOG(info,"hello" << idx_new);
													tmp(r1,r3) = tuple1.second(r1,r2)*tuple2.second(r2,r3);
													values[idx_new] = tmp[0];
													count++;
												}
											}

				}}}}}}}
				if (not finished)
					queue.push(std::pair<size_t,std::vector<std::vector<std::pair<std::vector<size_t>,Tensor>>>>(pos,data));

			}
				//XERUS_LOG(info, "count " << count);
		}
		size_t getIndex(size_t i1, size_t j1, size_t k1, size_t l1){
			return l1 + (p_down+1)*(k1 + (p_up+1)*(j1 + 3*i1));
		}

};

