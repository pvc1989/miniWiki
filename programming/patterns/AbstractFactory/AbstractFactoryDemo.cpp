#include <iostream>
#include <memory>

using std::cout;
using std::endl;
using std::unique_ptr;
using std::make_unique;

// AbstractProducts
struct AbstractProductA {
  virtual ~AbstractProductA() { }
  virtual void doA() = 0;
};
struct AbstractProductB {
  virtual ~AbstractProductB() { }
  virtual void doB() = 0;
};
// AbstractFactory
struct AbstractFactory {
  virtual ~AbstractFactory() { }
  virtual unique_ptr<AbstractProductA> CreateA() const = 0;
  virtual unique_ptr<AbstractProductB> CreateB() const = 0;
};
// Client uses only AbstractFactory and AbstractProducts.
struct Client {
  void use(const AbstractFactory& aFactory) {
    unique_ptr<AbstractProductA> aProductA = aFactory.CreateA();
    aProductA->doA();
    unique_ptr<AbstractProductB> aProductB = aFactory.CreateB();
    aProductB->doB();
  }
};
// ConreteProducts
struct ConcreteProductA1 : public AbstractProductA {
  void doA() override { cout << "ConcreteProductA1::doA()" << endl; }
};
struct ConcreteProductA2 : public AbstractProductA {
  void doA() override { cout << "ConcreteProductA2::doA()" << endl; }
};
struct ConcreteProductB1 : public AbstractProductB {
  void doB() override { cout << "ConcreteProductB1::doB()" << endl; }
};
struct ConcreteProductB2 : public AbstractProductB {
  void doB() override { cout << "ConcreteProductB2::doB()" << endl; }
};
// ConcreteFactories
struct ConcreteFactoryA1B1 : public AbstractFactory {
 public:
  unique_ptr<AbstractProductA> CreateA() const override {
    return make_unique<ConcreteProductA1>();
  }
  unique_ptr<AbstractProductB> CreateB() const override {
    return make_unique<ConcreteProductB1>();
  }
};
struct ConcreteFactoryA1B2 : public AbstractFactory {
 public:
  unique_ptr<AbstractProductA> CreateA() const override {
    return make_unique<ConcreteProductA1>();
  }
  unique_ptr<AbstractProductB> CreateB() const override {
    return make_unique<ConcreteProductB2>();
  }
};
struct ConcreteFactoryA2B1 : public AbstractFactory {
 public:
  virtual unique_ptr<AbstractProductA> CreateA() const override {
    return make_unique<ConcreteProductA2>();
  }
  virtual unique_ptr<AbstractProductB> CreateB() const override {
    return make_unique<ConcreteProductB1>();
  }
};
struct ConcreteFactoryA2B2 : public AbstractFactory {
 public:
  virtual unique_ptr<AbstractProductA> CreateA() const override {
    return make_unique<ConcreteProductA2>();
  }
  virtual unique_ptr<AbstractProductB> CreateB() const override {
    return make_unique<ConcreteProductB2>();
  }
};
// Choose a ConcreteFactory at run-time.
void prompt() {
  cout << "Usage:" << endl;
  cout << "  ./AbstractFactoryDemo.exe n" << endl;
  cout << "where n must be 11 or 12 or 21 or 22." << endl;
}
int main(int argc, const char* argv[]) {
  if (argc < 2) {
    prompt();
    return 0;
  }
  int n = std::stoi(argv[1]);
  unique_ptr<AbstractFactory> aFactory;
  switch (n) {
    case 11:
      aFactory.reset(new ConcreteFactoryA1B1());
      break;
    case 12:
      aFactory.reset(new ConcreteFactoryA1B2());
      break;
    case 21:
      aFactory.reset(new ConcreteFactoryA2B1());
      break;
    case 22:
      aFactory.reset(new ConcreteFactoryA2B2());
      break;
    default:
      prompt();
      return 0;
  }
  auto aProductA = aFactory->CreateA();
  aProductA->doA();
  aFactory->CreateB()->doB();
}
