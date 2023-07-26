public class Demo {
  // AbstractProducts
  public interface AbstractProductA {
    public abstract void doA();
  }
  public interface AbstractProductB {
    public abstract void doB();
  }
  // AbstractFactory
  public interface AbstractFactory {
    public abstract AbstractProductA CreateA();
    public abstract AbstractProductB CreateB();
  }
  // Client uses only AbstractFactory and AbstractProducts.
  public static class Client {
    public void use(AbstractFactory aFactory) {
      AbstractProductA aProductA = aFactory.CreateA();
      aProductA.doA();
      AbstractProductB aProductB = aFactory.CreateB();
      aProductB.doB();
    }
  }
  // ConcreteProducts
  public static class ConcreteProductA1 implements AbstractProductA {
    @Override
    public void doA() { System.out.println("aConcreteProductA1.doA()"); }
  }
  public static class ConcreteProductA2 implements AbstractProductA {
    @Override
    public void doA() { System.out.println("aConcreteProductA2.doA()"); }
  }
  public static class ConcreteProductB1 implements AbstractProductB {
    @Override
    public void doB() { System.out.println("aConcreteProductB1.doB()"); }
  }
  public static class ConcreteProductB2 implements AbstractProductB {
    @Override
    public void doB() { System.out.println("aConcreteProductB2.doB()"); }
  }
  // ConcreteFactories
  public static class ConcreteFactoryA1B1 implements AbstractFactory {
    @Override
    public AbstractProductA CreateA() { return new ConcreteProductA1(); }
    @Override
    public AbstractProductB CreateB() { return new ConcreteProductB1(); }
  }
  public static class ConcreteFactoryA1B2 implements AbstractFactory {
    @Override
    public AbstractProductA CreateA() { return new ConcreteProductA1(); }
    @Override
    public AbstractProductB CreateB() { return new ConcreteProductB2(); }
  }
  public static class ConcreteFactoryA2B1 implements AbstractFactory {
    @Override
    public AbstractProductA CreateA() { return new ConcreteProductA2(); }
    @Override
    public AbstractProductB CreateB() { return new ConcreteProductB1(); }
  }
  public static class ConcreteFactoryA2B2 implements AbstractFactory {
    @Override
    public AbstractProductA CreateA() { return new ConcreteProductA2(); }
    @Override
    public AbstractProductB CreateB() { return new ConcreteProductB2(); }
  }
  // Choose a ConcreteFactory at run-time.
  public static void prompt() {
    System.out.println("Usage:");
    System.out.println("  ./AbstractFactoryDemo.exe n");
    System.out.println("where n must be 11 or 12 or 21 or 22.");
  }
  public static void main(String[] args) {
    Client aClient = new Client();
    if (args.length < 1) {
      prompt();
      return;
    }
    int i = Integer.parseInt(args[0]);
    switch (i) {
      case 11:
        aClient.use(new ConcreteFactoryA1B1());
        break;
      case 12:
        aClient.use(new ConcreteFactoryA1B2());
        break;
      case 21:
        aClient.use(new ConcreteFactoryA2B1());
        break;
      case 22:
        aClient.use(new ConcreteFactoryA2B2());
        break;
      default:
        prompt();
        return;
    }
  }
}
