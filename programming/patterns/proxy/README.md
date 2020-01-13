# Proxy (Surrogate)

## 意图
Provide a surrogate or placeholder for another object to control access to it.

为一个（远程、重型、脆弱）对象提供代理或占位符，以便控制对该对象的访问。

## 用途
- **远程代理 (Remote Proxy)** 为远程对象提供一个本地代表。
- **虚拟代理 (Virtual Proxy)** 延迟创建重型对象。
- **保护代理 (Protection Proxy)** 限制对原始（脆弱）对象的访问。
- **智能引用 (Smart Reference)** 对裸指针进行封装，以便支持引用计数、动态资源管理等功能。

## 类图
[![](./class.svg)](./class.txt)

## 参与者
- **代理** (`Proxy`)
  - 维护对 `RealSubject` 实例的引用。
  - 实现与 `Subject` 相同的接口，以便替换 `RealSubject`。
  - 控制对 `RealSubject` 实例的访问，可能还需要负责创建、销毁 `RealSubject` 实例。
- **主体** (`Subject`)
  - 定义 `Proxy` 和 `RealSubject` 的公共接口，以便用 `Proxy` 实例替换 `RealSubject` 实例。
- **真实主体** (`RealSubject`)
  - 定义 `Proxy` 所表示的真实对象。
