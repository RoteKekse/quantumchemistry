#include <xerus.h>
#include <stdlib.h>
#include "contractpsihek.cpp"
#include "unitvectorprojection.cpp"
#include "../classes/containerhash.cpp"
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

			for (size_t pos=d-1; pos>position;pos--){
				tmp = xbase.first.get_component(pos);
				if (std::find(sample.begin(), sample.end(), pos) != sample.end())
					tmp.fix_mode(1,1);
				else
					tmp.fix_mode(1,0);
				rightxStack(i1) = tmp(i1,j1)*rightxStack(j1);
			}
			for (size_t pos=0; pos<position;pos++){
				tmp = xbase.second.get_component(pos);
				if (std::find(sample.begin(), sample.end(), pos) != sample.end())
					tmp.fix_mode(1,1);
				else
					tmp.fix_mode(1,0);
				leftxStack(i1) = tmp(j1,i1)*leftxStack(j1);
			}

			// projection
			if (std::find(sample.begin(), sample.end(), position) != sample.end())
				unit = Tensor::dirac({2},1);
			else
				unit = Tensor::dirac({2},0);

			comp(i1,i2,i3) = leftxStack(i1) * unit(i2)  * rightxStack(i3);

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

	public:
		PsiProbabilityFunction(TTTensor _psi): d(_psi.order()){
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

		std::vector<size_t> makeIndex(std::vector<size_t> sample){
			std::vector<size_t> index(d, 0);
			for (size_t i : sample)
				if (i < d)
					index[i] = 1;
			return index;
		}

};

