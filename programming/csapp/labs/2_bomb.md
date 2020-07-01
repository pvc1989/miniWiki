# Bomb Lab

## Dangerous Functions

### `explode_bomb()`

```c
void explode_bomb();
```

👉 Set a breakpoint at the head of this function to avoid explosion.

## Phase 1

### `string_length()`

```c
int string_length(char* s);
```

### `strings_not_equal()`

```nasm
Dump of assembler code for function strings_not_equal:
   0x401338 <+0>:     push   %r12
   0x40133a <+2>:     push   %rbp
   0x40133b <+3>:     push   %rbx
   0x40133c <+4>:     mov    %rdi,%rbx
   0x40133f <+7>:     mov    %rsi,%rbp
   0x401342 <+10>:    callq  0x40131b <string_length>  # l1 = string_length(s1);
   0x401347 <+15>:    mov    %eax,%r12d
   0x40134a <+18>:    mov    %rbp,%rdi
   0x40134d <+21>:    callq  0x40131b <string_length>  # l2 = string_length(s2);
   0x401352 <+26>:    mov    $0x1,%edx
   0x401357 <+31>:    cmp    %eax,%r12d  # if (l1 == l2)
   0x40135a <+34>:    jne    0x40139b <strings_not_equal+99>  # else
   0x40135c <+36>:    movzbl (%rbx),%eax  # R[%al] = s1[0]
   0x40135f <+39>:    test   %al,%al  # if(s1[0] == '\0')
   0x401361 <+41>:    je     0x401388 <strings_not_equal+80>  # then
                      # else, i.e. (s1[0] != '\0')
   0x401363 <+43>:    cmp    0x0(%rbp),%al  # if (*s1 == *s2)
   0x401366 <+46>:    je     0x401372 <strings_not_equal+58>  # then
   0x401368 <+48>:    jmp    0x40138f <strings_not_equal+87>  # else
   0x40136a <+50>:    cmp    0x0(%rbp),%al  # if (*s1 == *s2)
   0x40136d <+53>:    nopl   (%rax)  # no operation
   0x401370 <+56>:    jne    0x401396 <strings_not_equal+94>  # else
   0x401372 <+58>:    add    $0x1,%rbx  # ++s1
   0x401376 <+62>:    add    $0x1,%rbp  # ++s2
   0x40137a <+66>:    movzbl (%rbx),%eax  # R[%al] = *s1
   0x40137d <+69>:    test   %al,%al  # if (*s1 == `\0`)
   0x40137f <+71>:    jne    0x40136a <strings_not_equal+50>  # else
                      # return 0;
   0x401381 <+73>:    mov    $0x0,%edx  # *s1 == `\0`
   0x401386 <+78>:    jmp    0x40139b <strings_not_equal+99>
   0x401388 <+80>:    mov    $0x0,%edx  # s1[0] == '\0'
   0x40138d <+85>:    jmp    0x40139b <strings_not_equal+99>
                      # return 1;
   0x40138f <+87>:    mov    $0x1,%edx  # *s1 != *s2
   0x401394 <+92>:    jmp    0x40139b <strings_not_equal+99>
   0x401396 <+94>:    mov    $0x1,%edx  # temp = 1;
   0x40139b <+99>:    mov    %edx,%eax  # result = temp;
   0x40139d <+101>:   pop    %rbx
   0x40139e <+102>:   pop    %rbp
   0x40139f <+103>:   pop    %r12
   0x4013a1 <+105>:   retq
```

```c
int strings_not_equal(char* s1, char* s2) {  /* 0x401338 */
  int l1 = string_length(s1);  /* <+10> */
  int l2 = string_length(s2);  /* <+21> */
  if (l1 == l2) {  /* <+31> */
    while (*s1 != '\0') {  /* <+39> <+69> */
      if (*s1 == *s2) {  /* <+50> */
        ++s1;  /* <+58> */
        ++s2;  /* <+62> */
      } else {
        return 1;/* <+87> */
      }
    }
    return 0;  /* <+73> <+80> */
  } else {
    return 1;  /* <+99> */
  }
}
```

### `phase_1()`

```nasm
Dump of assembler code for function phase_1:
   0x400ee0 <+0>:     sub    $0x8,%rsp
   0x400ee4 <+4>:     mov    $0x402400,%esi
   0x400ee9 <+9>:     callq  0x401338 <strings_not_equal>
   0x400eee <+14>:    test   %eax,%eax
   0x400ef0 <+16>:    je     0x400ef7 <phase_1+23>
   0x400ef2 <+18>:    callq  0x40143a <explode_bomb>
   0x400ef7 <+23>:    add    $0x8,%rsp
   0x400efb <+27>:    retq 
```

```c
void phase_1(char* input) {  /* 0x400ee0 */
  /* run `x/s 0x402400` in gdb */
  char* key = "Border relations with Canada have never been better.";
  if (strings_not_equal(input/* %rdi */, key/* %rsi */)) {
    explode_bomb();
  } else {
    return;
  }
}
```

So, the 1st line should be

```
Border relations with Canada have never been better.
```

## Phase 2

### `read_six_numbers()`

```nasm
Dump of assembler code for function read_six_numbers:
   0x40145c <+0>:     sub    $0x18,%rsp
   0x401460 <+4>:     mov    %rsi,%rdx
   0x401463 <+7>:     lea    0x4(%rsi),%rcx
   0x401467 <+11>:    lea    0x14(%rsi),%rax
   0x40146b <+15>:    mov    %rax,0x8(%rsp)
   0x401470 <+20>:    lea    0x10(%rsi),%rax
   0x401474 <+24>:    mov    %rax,(%rsp)
   0x401478 <+28>:    lea    0xc(%rsi),%r9
   0x40147c <+32>:    lea    0x8(%rsi),%r8
   0x401480 <+36>:    mov    $0x4025c3,%esi   # "%d %d %d %d %d %d"
   0x401485 <+41>:    mov    $0x0,%eax        # n = 0
   0x40148a <+46>:    callq  0x400bf0 <__isoc99_sscanf@plt>
   0x40148f <+51>:    cmp    $0x5,%eax  # n - 5
   0x401492 <+54>:    jg     0x401499 <read_six_numbers+61>  # n-5 > 0
   0x401494 <+56>:    callq  0x40143a <explode_bomb>         # n-5 <= 0 
   0x401499 <+61>:    add    $0x18,%rsp
   0x40149d <+65>:    retq
```

```c
int read_six_numbers(char* s, int a[]);
```

### `phase_2()`

```nasm
Dump of assembler code for function phase_2:
   0x400efc <+0>:     push   %rbp
   0x400efd <+1>:     push   %rbx
   0x400efe <+2>:     sub    $0x28,%rsp  # allocate local array
   0x400f02 <+6>:     mov    %rsp,%rsi   # head of the array
   0x400f05 <+9>:     callq  0x40145c <read_six_numbers>
                      # use `x/6wd $rsp` to examine these numbers
   0x400f0a <+14>:    cmpl   $0x1,(%rsp)  # a[0] - 1
   0x400f0e <+18>:    je     0x400f30 <phase_2+52>    # == 0
   0x400f10 <+20>:    callq  0x40143a <explode_bomb>  # != 0
   0x400f15 <+25>:    jmp    0x400f30 <phase_2+52>
   0x400f17 <+27>:    mov    -0x4(%rbx),%eax  # R[%eax] = a[i-1]
   0x400f1a <+30>:    add    %eax,%eax        # R[%eax] = a[i-1]*2
   0x400f1c <+32>:    cmp    %eax,(%rbx)      # a[i] - a[i-1]*2
   0x400f1e <+34>:    je     0x400f25 <phase_2+41>    # == 0
   0x400f20 <+36>:    callq  0x40143a <explode_bomb>  # != 0
   0x400f25 <+41>:    add    $0x4,%rbx        # ++i
   0x400f29 <+45>:    cmp    %rbp,%rbx        # &a[i] - &a[5]
   0x400f2c <+48>:    jne    0x400f17 <phase_2+27>  # != 0
   0x400f2e <+50>:    jmp    0x400f3c <phase_2+64>  # == 0
   0x400f30 <+52>:    lea    0x4(%rsp),%rbx   # R[%rbx] = &a[1]
   0x400f35 <+57>:    lea    0x18(%rsp),%rbp  # R[%rbp] = &a[5]
   0x400f3a <+62>:    jmp    0x400f17 <phase_2+27>
   0x400f3c <+64>:    add    $0x28,%rsp
   0x400f40 <+68>:    pop    %rbx
   0x400f41 <+69>:    pop    %rbp
   0x400f42 <+70>:    retq
```

```c
void phase_2(char* input) {  /* 0x400efc */
  int a[6];  /* <+2> */
  int n = read_six_numbers(input, a);  /* <+9> */
  if (a[0] != 1) {  /* <+14> */
    explode_bomb();
  }
  int* curr = &a[1];  /* <+52> */
  int* last = &a[5];  /* <+57> */
  do {
    int value = curr[-1];  /* <+27> */
    value += value;        /* <+30> */
    if (*curr != value) {  /* <+32> */
      explode_bomb();
    }
    ++curr;  /* <+41> */
  } while (curr != last);  /* <+45> */
}
```

So, the 2nd line should begin with

```
1 2 4 8 16 32
```

## Phase 3

### `phase_3()`

```nasm
Dump of assembler code for function phase_3:
   0x400f43 <+0>:     sub    $0x18,%rsp
   0x400f47 <+4>:     lea    0xc(%rsp),%rcx  # int y
   0x400f4c <+9>:     lea    0x8(%rsp),%rdx  # int x
   0x400f51 <+14>:    mov    $0x4025cf,%esi  # "%d %d"
   0x400f56 <+19>:    mov    $0x0,%eax
   0x400f5b <+24>:    callq  0x400bf0 <__isoc99_sscanf@plt>
   0x400f60 <+29>:    cmp    $0x1,%eax  # n - 1
   0x400f63 <+32>:    jg     0x400f6a <phase_3+39>    # > 0
   0x400f65 <+34>:    callq  0x40143a <explode_bomb>  # <= 0
   0x400f6a <+39>:    cmpl   $0x7,0x8(%rsp)  # x - 7
   0x400f6f <+44>:    ja     0x400fad <phase_3+106>  # > 0
   0x400f71 <+46>:    mov    0x8(%rsp),%eax          # <= 0 (unsigned)
   0x400f75 <+50>:    jmpq   *0x402470(,%rax,8)  # switch (x)
   # use `x/8gx 0x402470` to examine the table, which gives
   #   0x00402470: 0x0000000000400f7c 0x0000000000400fb9
   #   0x00402480: 0x0000000000400f83 0x0000000000400f8a
   #   0x00402490: 0x0000000000400f91 0x0000000000400f98
   #   0x004024a0: 0x0000000000400f9f 0x0000000000400fa6
   # thus we have
   # case 0:
   0x400f7c <+57>:    mov    $0xcf,%eax   # t = 207
   0x400f81 <+62>:    jmp    0x400fbe <phase_3+123>
   # case 2:
   0x400f83 <+64>:    mov    $0x2c3,%eax  # t = 707
   0x400f88 <+69>:    jmp    0x400fbe <phase_3+123>
   # case 3:
   0x400f8a <+71>:    mov    $0x100,%eax  # t = 256
   0x400f8f <+76>:    jmp    0x400fbe <phase_3+123>
   # case 4:
   0x400f91 <+78>:    mov    $0x185,%eax  # t = 389
   0x400f96 <+83>:    jmp    0x400fbe <phase_3+123>
   # case 5:
   0x400f98 <+85>:    mov    $0xce,%eax   # t = 206
   0x400f9d <+90>:    jmp    0x400fbe <phase_3+123>
   # case 6:
   0x400f9f <+92>:    mov    $0x2aa,%eax  # t = 682
   0x400fa4 <+97>:    jmp    0x400fbe <phase_3+123>
   # case 7:
   0x400fa6 <+99>:    mov    $0x147,%eax  # t = 327
   0x400fab <+104>:   jmp    0x400fbe <phase_3+123>
   # default:
   0x400fad <+106>:   callq  0x40143a <explode_bomb>  # (unsigned)x > 7
   0x400fb2 <+111>:   mov    $0x0,%eax    # t = 0
   0x400fb7 <+116>:   jmp    0x400fbe <phase_3+123>
   # case 1:
   0x400fb9 <+118>:   mov    $0x137,%eax  # t = 311
   # out of the switch
   0x400fbe <+123>:   cmp    0xc(%rsp),%eax  # t - y
   0x400fc2 <+127>:   je     0x400fc9 <phase_3+134>   # == 0
   0x400fc4 <+129>:   callq  0x40143a <explode_bomb>  # != 0
   0x400fc9 <+134>:   add    $0x18,%rsp
   0x400fcd <+138>:   retq 
```

```c
void phase_3(char* input) {
  unsigned int x
  unsigned int y;
  int n = sscanf(input, "%d %d", &x, &y);
  if (n <= 1) {
    explode_bomb();  /* <+34> */
  }
  unsigned int t;  /* %rax */
  switch (x) {  /* <+39> */
  case 0:
    t = 0xcf;
    break;
  case 1:
    t = 0x137;
    break;
  case 2:
    t = 0x2c3;
    break;
  case 3:
    t = 0x100;
    break;
  case 4:
    t = 0x185;
    break;
  case 5:
    t = 0xce;
    break;
  case 6:
    t = 0x2aa;
    break;
  case 7:
    t = 0x147;
    break;
  default:
    explode_bomb();  /* <+106> */
    t = 0;
    break;
  }
  if (t != y) {  /* <+123> */
    explode_bomb();  /* <+129> */
  }
}
```

So, the 3rd line should begin with any one of the following 8 cases:

```
0 207
1 311
2 707
3 389
4 206
5 682
6 682
7 327
```

## Phase 4

### `func4()`

```nasm
Dump of assembler code for function func4:
   0x400fce <+0>:     sub    $0x8,%rsp
   0x400fd2 <+4>:     mov    %edx,%eax   # t = c
   0x400fd4 <+6>:     sub    %esi,%eax   # t -= b
   0x400fd6 <+8>:     mov    %eax,%ecx   # s = t
   0x400fd8 <+10>:    shr    $0x1f,%ecx  # s / (1<<31)
   0x400fdb <+13>:    add    %ecx,%eax   # t += s
   0x400fdd <+15>:    sar    %eax        # t /= 2
   0x400fdf <+17>:    lea    (%rax,%rsi,1),%ecx  # s = t + b
   0x400fe2 <+20>:    cmp    %edi,%ecx   # s - a
   0x400fe4 <+22>:    jle    0x400ff2 <func4+36>
                      # s > a
   0x400fe6 <+24>:    lea    -0x1(%rcx),%edx
   0x400fe9 <+27>:    callq  0x400fce <func4>  # t = func4(a, b, s-1)
   0x400fee <+32>:    add    %eax,%eax         # t += t
   0x400ff0 <+34>:    jmp    0x401007 <func4+57>
                      # s <= a
   0x400ff2 <+36>:    mov    $0x0,%eax  # t = 0
   0x400ff7 <+41>:    cmp    %edi,%ecx  # s - a
   0x400ff9 <+43>:    jge    0x401007 <func4+57>
   0x400ffb <+45>:    lea    0x1(%rcx),%esi
   0x400ffe <+48>:    callq  0x400fce <func4>  # t = func4(a, s+1, c)
   0x401003 <+53>:    lea    0x1(%rax,%rax,1),%eax  # t = 1 + 2*t
                      # return t
   0x401007 <+57>:    add    $0x8,%rsp
   0x40100b <+61>:    retq 
```

```c
int func4(int a, int b, int c) {
  int func4(int a, int b, int c) {
  int t = c - b;
  int s = t;
  s /= (1<<31);
  t += s;
  t /= 2;
  s = t + b;
  if (s > a) {
    t = func4(a, b, s-1);
    t += t;
  } else {
    t = 0;
    if (s < a) {
      t = func4(a, s+1, c);
      t = 1 + t + t;
    }
  }
  return t;
}
```

### `phase_4()`

```nasm
Dump of assembler code for function phase_4:
   0x40100c <+0>:     sub    $0x18,%rsp
   0x401010 <+4>:     lea    0xc(%rsp),%rcx
   0x401015 <+9>:     lea    0x8(%rsp),%rdx
   0x40101a <+14>:    mov    $0x4025cf,%esi  # "%d %d"
   0x40101f <+19>:    mov    $0x0,%eax
   0x401024 <+24>:    callq  0x400bf0 <__isoc99_sscanf@plt>
   0x401029 <+29>:    cmp    $0x2,%eax  # n - 2
   0x40102c <+32>:    jne    0x401035 <phase_4+41>  # n != 2
   0x40102e <+34>:    cmpl   $0xe,0x8(%rsp)           # x - 0xe
   0x401033 <+39>:    jbe    0x40103a <phase_4+46>
   0x401035 <+41>:    callq  0x40143a <explode_bomb>  # x > 0xe
   0x40103a <+46>:    mov    $0xe,%edx
   0x40103f <+51>:    mov    $0x0,%esi
   0x401044 <+56>:    mov    0x8(%rsp),%edi
   0x401048 <+60>:    callq  0x400fce <func4>  # z = func4(x, 0, 14)
   0x40104d <+65>:    test   %eax,%eax  # z & z
   0x40104f <+67>:    jne    0x401058 <phase_4+76>  # z != 0
   0x401051 <+69>:    cmpl   $0x0,0xc(%rsp)  # y - 0
   0x401056 <+74>:    je     0x40105d <phase_4+81>    # == 0
   0x401058 <+76>:    callq  0x40143a <explode_bomb>  # != 0
   0x40105d <+81>:    add    $0x18,%rsp
   0x401061 <+85>:    retq   
```

```c
void phase_4(char* input) {
  unsigned int x, y;
  unsigned int n = sscanf(input, "%d %d", &x, &y);
  if (n != 2 || x > 14) {
    explode_bomb();  /* <+41> */
  }
  int z = func4(x, 0, 14);  /* <+60> */
  if (z || y) {  /* <+65> <+69> */
    explode_bomb();  /* <+76> */
  }
}
```

👉 It is not easy to find a formula in closed form for the recursive function `func4()`. However, we only need ***one*** `x` such that `func4(x, 0, 14) == 0`. So, the 4th line should begin with any one of the following cases:

```
0 0
1 0
3 0
```

## Phase 5

### `phase_5()`

```nasm
Dump of assembler code for function phase_5:
   0x401062 <+0>:     push   %rbx
   0x401063 <+1>:     sub    $0x20,%rsp
   0x401067 <+5>:     mov    %rdi,%rbx
   0x40106a <+8>:     mov    %fs:0x28,%rax  # store canary
   0x401073 <+17>:    mov    %rax,0x18(%rsp)
   0x401078 <+22>:    xor    %eax,%eax
   0x40107a <+24>:    callq  0x40131b <string_length>
   0x40107f <+29>:    cmp    $0x6,%eax
   0x401082 <+32>:    je     0x4010d2 <phase_5+112>  # wanted
   0x401084 <+34>:    callq  0x40143a <explode_bomb>
   0x401089 <+39>:    jmp    0x4010d2 <phase_5+112>
   # \begin{loop}
   0x40108b <+41>:    movzbl (%rbx,%rax,1),%ecx  # int R[%ecx] = a[i]
   0x40108f <+45>:    mov    %cl,(%rsp)   # char M[R[%rsp]] = a[i]
   0x401092 <+48>:    mov    (%rsp),%rdx  # long R[%rdx] = M[R[%rsp]]
   0x401096 <+52>:    and    $0xf,%edx    # a[i] is the loweset byte
   0x401099 <+55>:    movzbl 0x4024b0(%rdx),%edx  # s = M[0x4024b0 + s]
   0x4010a0 <+62>:    mov    %dl,0x10(%rsp,%rax,1)  # a[i] = s
   0x4010a4 <+66>:    add    $0x1,%rax  # ++i
   0x4010a8 <+70>:    cmp    $0x6,%rax  # i - 6
   0x4010ac <+74>:    jne    0x40108b <phase_5+41>  # while (i != 0)
   # \end{loop}
   0x4010ae <+76>:    movb   $0x0,0x16(%rsp)  # a[6] == `\0`
   0x4010b3 <+81>:    mov    $0x40245e,%esi   # "flyers" as 2nd arg
   0x4010b8 <+86>:    lea    0x10(%rsp),%rdi  #   &a[0]  as 1st arg
   0x4010bd <+91>:    callq  0x401338 <strings_not_equal>
   0x4010c2 <+96>:    test   %eax,%eax
   0x4010c4 <+98>:    je     0x4010d9 <phase_5+119>  # wanted
   0x4010c6 <+100>:   callq  0x40143a <explode_bomb>
   0x4010cb <+105>:   nopl   0x0(%rax,%rax,1)
   0x4010d0 <+110>:   jmp    0x4010d9 <phase_5+119>
   0x4010d2 <+112>:   mov    $0x0,%eax
   0x4010d7 <+117>:   jmp    0x40108b <phase_5+41>
   0x4010d9 <+119>:   mov    0x18(%rsp),%rax  # pick out stored canary
   0x4010de <+124>:   xor    %fs:0x28,%rax
   0x4010e7 <+133>:   je     0x4010ee <phase_5+140>  # no overflow
   0x4010e9 <+135>:   callq  0x400b30 <__stack_chk_fail@plt>
   0x4010ee <+140>:   add    $0x20,%rsp  # good bye
   0x4010f2 <+144>:   pop    %rbx
   0x4010f3 <+145>:   retq 
```

### Step-by-Step

1. From `<+24>` to `<+34>`, we know ***Line 5 should be a 6-`char` string (excluding the `'\0'` at the end)***.

1. There is a 6-step loop between `<+41>` and `<+74>`, which does not call `explode_bomb()`. So, it's safe to set a breakpoint after it, and let the process keep running until `<+76>` is hit.

1. From `<+76>` to `<+100>`, we know ***the target string at `0x40245e` is `flyers`***, which is indeed 6-`char` long.

1. Now, it's time to examine the loop in detail. We are working on a *little-endian* system, so `a[i]` resides in the least-significant byte, and the the net effect of the 3 instructions beginning at `<+81>` is
   ```c
   R[%rdx] = ZeroExtend(a[i] % 0x10);
   ```
   So, the C code for the loop might be
   ```c
   void phase_5(char* input) {
     if (string_length(input) != 6) explode_bomb();
     for (char* curr = a; curr != a+6; ++curr) {
       int t = *curr;    /* %ecx */
       long s = t % 16;  /* %rdx */
       s = *(int *)(0x4024b0 + s);
       *curr = (char) s;
     }
   }
   ```
   
1. The 16 bytes beginning at `0x4024b0`, denoted as `char b[16]`, are
   ```shell
   (lldb) x/16bd 0x4024b0
   0x004024b0: 109
   0x004024b1: 97
   0x004024b2: 100
   0x004024b3: 117
   0x004024b4: 105
   0x004024b5: 101
   0x004024b6: 114
   0x004024b7: 115
   0x004024b8: 110
   0x004024b9: 102
   0x004024ba: 111
   0x004024bb: 116
   0x004024bc: 118
   0x004024bd: 98
   0x004024be: 121
   0x004024bf: 108
   ```
while the 6 bytes in `flyers`, beginning at `0x40245e`, denoted as `char c[6]`, are
   ```shell
   (lldb) x/6bd 0x40245e
   0x0040245e: 102
   0x0040245f: 108
   0x00402460: 121
   0x00402461: 101
   0x00402462: 114
   0x00402463: 115
   ```
   So, the mapping gives a strong implication to the answer:
   ```c
   c[0] == 102 == b[0x9]
   c[1] == 108 == b[0xf]
   c[2] == 121 == b[0xe]
   c[3] == 101 == b[0x5]
   c[4] == 114 == b[0x6]
   c[5] == 115 == b[0x7]
   ```
   
1. The 6 bytes in Line 5, denoting as `char a[6]`, should be
   ```c
   a[0] == 0x9 + 0x10*K
   a[1] == 0xf + 0x10*K
   a[2] == 0xe + 0x10*K
   a[3] == 0x5 + 0x10*K
   a[4] == 0x6 + 0x10*K
   a[5] == 0x7 + 0x10*K
   ```
   The answer is not unique. When `k == 4` the answer is
   ```c
   0x49 == 'I'
   0x4f == 'O'
   0x4e == 'N'
   0x45 == 'E'
   0x46 == 'F'
   0x47 == 'G'
   ```

## Phase 6

### `read_six_numbers()`

```nasm
0x4010f4 <+0>:     push   %r14
0x4010f6 <+2>:     push   %r13
0x4010f8 <+4>:     push   %r12
0x4010fa <+6>:     push   %rbp
0x4010fb <+7>:     push   %rbx
0x4010fc <+8>:     sub    $0x50,%rsp  # allocate array
0x401100 <+12>:    mov    %rsp,%r13
0x401103 <+15>:    mov    %rsp,%rsi
0x401106 <+18>:    callq  0x40145c <read_six_numbers>
0x40110b <+23>:    mov    %rsp,%r14  # save &a[0] for later use
```
`read_six_numbers()` implies ***Line 6 must begin with 6 integers***. So, the first 2 lines in the C code looks like

```c
int a[6];
read_six_numbers(input, a);  /* <+18> */
```

### `1, 2, 3, 4, 5, 6`

```nasm
# init the loop
0x40110e <+26>:    mov    $0x0,%r12d  # int i = 0
# enter the loop
0x401114 <+32>:    mov    %r13,%rbp  # R[%rbp] == &a[i]
0x401117 <+35>:    mov    0x0(%r13),%eax  # unsigned int t = a[i]
0x40111b <+39>:    sub    $0x1,%eax  # t -= 1
0x40111e <+42>:    cmp    $0x5,%eax  # t - 5
0x401121 <+45>:    jbe    0x401128 <phase_6+52>  # unsigned <=
0x401123 <+47>:    callq  0x40143a <explode_bomb>  # t > 5
0x401128 <+52>:    add    $0x1,%r12d  # ++i
0x40112c <+56>:    cmp    $0x6,%r12d  # i - 6
0x401130 <+60>:    je     0x401153 <phase_6+95>  # i == 6
# enter the inner loop
0x401132 <+62>:    mov    %r12d,%ebx  # int j = i
0x401135 <+65>:    movslq %ebx,%rax   # R[%rax] = j
0x401138 <+68>:    mov    (%rsp,%rax,4),%eax  # R[%eax] = a[j]
0x40113b <+71>:    cmp    %eax,0x0(%rbp)  # a[i-1] - a[j]
0x40113e <+74>:    jne    0x401145 <phase_6+81>
0x401140 <+76>:    callq  0x40143a <explode_bomb>  # a[i-1] == a[j]
0x401145 <+81>:    add    $0x1,%ebx  # ++j
0x401148 <+84>:    cmp    $0x5,%ebx  # j - 5
0x40114b <+87>:    jle    0x401135 <phase_6+65>  # j <= 5
# leave the inner loop
0x40114d <+89>:    add    $0x4,%r13  # a += 1
0x401151 <+93>:    jmp    0x401114 <phase_6+32>
# leave the loop
```

The `jbe` in `<+45>` tells us the temporary `int` in `%eax` should be `unsigned`.

The nested loops between `<+32>` and `<+93>` correspond to the following C code:
```c
for (int i = 0; true; ) {
  unsigned int t = a[i] - 1;
  if (t > 5) explode_bomb();  /* <+47> */
  ++i;
  if (i == 6) break;
  for (int j = i; j <= 5; ++j)
    if (a[i-1] == a[j])  /* <+76> */
      explode_bomb();
}
```
It tells us ***the 6 numbers must be distinct (implied by `<+76>`) and within `{1, 2, 3, 4, 5, 6}` (implied by `<+47>`)***; their order, however, is undetermined. Try the most natural one, for example:

```nasm
(lldb) x/6wd $rsp
0x7fffffffe4c0: 1
0x7fffffffe4c4: 2
0x7fffffffe4c8: 3
0x7fffffffe4cc: 4
0x7fffffffe4d0: 5
0x7fffffffe4d4: 6
```

### `a[i] = 7 - a[i]`

```nasm
# init the loop
0x401153 <+95>:    lea    0x18(%rsp),%rsi  # int* a_n = &a[6]
0x401158 <+100>:   mov    %r14,%rax        # int* a_i = &a[i]
0x40115b <+103>:   mov    $0x7,%ecx  # t = 7
# enter the loop
0x401160 <+108>:   mov    %ecx,%edx  # s = t
0x401162 <+110>:   sub    (%rax),%edx  # s -= a[i]
0x401164 <+112>:   mov    %edx,(%rax)  # a[i] = 7 - a[i] 👈
0x401166 <+114>:   add    $0x4,%rax  # a_i += 1
0x40116a <+118>:   cmp    %rsi,%rax  # a_i - a_n
0x40116d <+121>:   jne    0x401160 <phase_6+108>  # a_i != a_n
# leave the loop
```

This loop replaces `a[i]` with `7 - a[i]` for each `i`, that is

```c
int* a_i = a;
int* a_n = a + 6;
int t = 7;
do {
  *a_i = 7 - *a_i;
  a_i += 1;
} while (a_i != a_n);
```

So, it's safe to go through the loop and stop after it.

### Array of Nodes

```nasm
# init the loop
0x40116f <+123>:   mov    $0x0,%esi  # void* offset = 0
# begin loop
0x401174 <+128>:   jmp    0x401197 <phase_6+163>
# begin inner loop
0x401176 <+130>:   mov    0x8(%rdx),%rdx  # target = *(target + 8)
0x40117a <+134>:   add    $0x1,%eax  # count += 1
0x40117d <+137>:   cmp    %ecx,%eax  # count - value
0x40117f <+139>:   jne    0x401176 <phase_6+130>  # while (count != value)
0x401181 <+141>:   jmp    0x401188 <phase_6+148>
# end inner loop
0x401183 <+143>:   mov    $0x6032d0,%edx  # if (value <= 1)
# after if-else
0x401188 <+148>:   mov    %rdx,0x20(%rsp,%rsi,2)  # b[i] = target
0x40118d <+153>:   add    $0x4,%rsi   # offset += 4
0x401191 <+157>:   cmp    $0x18,%rsi  # offset - 24
0x401195 <+161>:   je     0x4011ab <phase_6+183>
# enter the loop
0x401197 <+163>:   mov    (%rsp,%rsi,1),%ecx  # int value = *(int*)((void*) a + offset)
0x40119a <+166>:   cmp    $0x1,%ecx  # value - 1
0x40119d <+169>:   jle    0x401183 <phase_6+143>  # if (value <= 1)
0x40119f <+171>:   mov    $0x1,%eax  # if (value > 1) count = 1
0x4011a4 <+176>:   mov    $0x6032d0,%edx  # void* target = 0x6032d0
0x4011a9 <+181>:   jmp    0x401176 <phase_6+130>  # inner loop
# leave the loop
```

This section is another loop:

```c
T* b[6];
void* offset = 0;  /* <+123> */
/* <+130> */
while (offset != 24/* <+157> */) {
  T* target;
  int value = *(int*)((void*)a + offset);  /* <+163> */
  if (value > 1) {  /* <+166> */
    /* <+171> */
    int count = 1;
    target = 0x6032d0;  /* <+176> */
    do {
      target = *(T*)((void*)target + 8); /* <+130> */
      ++count;
    } while (count != value);  /* <+139> */
  } else {  /* value <= 1 */
    target = 0x6032d0;  /* <+143> */
  }
  *((void*)b + offset*2) = target;  /* <+148> */
  offset += 4;
} /* <+181> */
```

From `<+130>` we know `T` must be a recursive data structure, e.g.

```c
struct T {
  long data;
  T* next;
};
```

By looking at the bytes started at `0x6032d0`, we know that it is a ***linked list***:

```nasm
(lldb) x/12gx 0x6032d0
0x006032d0: 0x000000010000014c 0x00000000006032e0
0x006032e0: 0x00000002000000a8 0x00000000006032f0
0x006032f0: 0x000000030000039c 0x0000000000603300
0x00603300: 0x00000004000002b3 0x0000000000603310
0x00603310: 0x00000005000001dd 0x0000000000603320
0x00603320: 0x00000006000001bb 0x0000000000000000
```

So, the loop can be clarified as

```c
T* b[6];
for (int i = 0; i != 6; ++i) {
  T* target = 0x6032d0;
  int value = a[i];
  int count = 1;
  while (count != value) {
    target = target->next; /* <+130> */
    ++count;
  }
  b[i] = target;
}
```

Again, it's safe to go through the loop and stop after it (at `<+183>`).

The loop use `int a[6]` to build another array, located at `%rsp+0x20` and denoted as `T* b[6]`, whose elements come from the linked list and reordered by `a[]`'s elements':

```nasm
# stop at <+183>
(lldb) x/6wd $rsp # int a[6]
0x7fffffffe4c0: 6
0x7fffffffe4c4: 5
0x7fffffffe4c8: 4
0x7fffffffe4cc: 3
0x7fffffffe4d0: 2
0x7fffffffe4d4: 1
(lldb) x/6gx $rsp+0x20  # T* b[6]
0x7fffffffe4e0: 0x0000000000603320 0x0000000000603310
0x7fffffffe4f0: 0x0000000000603300 0x00000000006032f0
0x7fffffffe500: 0x00000000006032e0 0x00000000006032d0
```

### Rebuild the List
```nasm
0x4011ab <+183>:   mov    0x20(%rsp),%rbx  # T* rbx = b[0];  // head
0x4011b0 <+188>:   lea    0x28(%rsp),%rax  # T** rax = &b[1];
0x4011b5 <+193>:   lea    0x50(%rsp),%rsi  # T** rsi = &b[6];  // tail
0x4011ba <+198>:   mov    %rbx,%rcx        # T* rcx = b[0];  // b[i-1]
# enter the loop
0x4011bd <+201>:   mov    (%rax),%rdx      # T* rdx = *rax;  // b[i]
0x4011c0 <+204>:   mov    %rdx,0x8(%rcx)   # rcx->next = rdx;
0x4011c4 <+208>:   add    $0x8,%rax        # rax += 1;  // &b[i+1]
0x4011c8 <+212>:   cmp    %rsi,%rax        # rax - tail
0x4011cb <+215>:   je     0x4011d2 <phase_6+222> 
0x4011cd <+217>:   mov    %rdx,%rcx        # rcx = rdx;  // b[i]
0x4011d0 <+220>:   jmp    0x4011bd <phase_6+201>  # while (rax != tail)
# leave the loop
```
This loop just rebuild a linked list according to the new order:
```c
T** curr = &b[1];
T** tail = &b[6];
T*  prev =  b[0];
while (curr != tail) {
  prev->next = *curr;
  curr += 1;
  prev = *curr;
}
```

### Sort by `rank`

At this point, `%rdx` holds `b[5]` and `%rbx` holds `b[0]`.

```nasm
0x4011d2 <+222>:   movq   $0x0,0x8(%rdx)  # b[5]->next = 0;
0x4011da <+230>:   mov    $0x5,%ebp  # i = 5;
# enter the loop
0x4011df <+235>:   mov    0x8(%rbx),%rax  # rax = rbx->next;
0x4011e3 <+239>:   mov    (%rax),%eax     # int eax = *rax;
0x4011e5 <+241>:   cmp    %eax,(%rbx)     # rbx->data - eax;
0x4011e7 <+243>:   jge    0x4011ee <phase_6+250>
0x4011e9 <+245>:   callq  0x40143a <explode_bomb>  # rbx->data < eax
0x4011ee <+250>:   mov    0x8(%rbx),%rbx  # rbx = rbx->next;
0x4011f2 <+254>:   sub    $0x1,%ebp  # --i;
0x4011f5 <+257>:   jne    0x4011df <phase_6+235>
# leave the loop
```

This loop checks each pair of neighbouring nodes:

```c
b[5]->next = 0;  /* <+222> */
T* curr = b;
int i = 5/* <+230> */;
while (i != 0/* <+257> */) {
  T* next = curr->next;  /* <+235> */
  if (curr->data < next->data)
    explode_bomb();  /* <+245> */
  curr = curr->next;  /* <+250> */
  --i;/* <+250> */
}
```

It is now clear that the `data` field should begin with an `int` member:
```c
struct node {
  int rank;
  int id;  /* optional */
  struct node* next;
};
```
and this member should keep decreasing along the list. Recall the original list:
```nasm
(lldb) x/12gx 0x6032d0
0x006032d0: 0x000000010000014c 0x00000000006032e0
0x006032e0: 0x00000002000000a8 0x00000000006032f0
0x006032f0: 0x000000030000039c 0x0000000000603300
0x00603300: 0x00000004000002b3 0x0000000000603310
0x00603310: 0x00000005000001dd 0x0000000000603320
0x00603320: 0x00000006000001bb 0x0000000000000000
```

It can be transformed into the following table:

|  Address   | `rank`  | `id` |   `next`   |
| :--------: | :-----: | :--: | :--------: |
| `0x6032d0` | `0x14c` | `1`  | `0x6032e0` |
| `0x6032e0` | `0x0a8` | `2`  | `0x6032f0` |
| `0x6032f0` | `0x39c` | `3`  | `0x603300` |
| `0x603300` | `0x2b3` | `4`  | `0x603310` |
| `0x603310` | `0x1dd` | `5`  | `0x603320` |
| `0x603320` | `0x1bb` | `6`  | `0x000000` |

The `rank`s are initially unsorted. Sort then in the descending order, we have

|  Address   | `rank`  | `id` |   `next`   |
| :--------: | :-----: | :--: | :--------: |
| `0x6032f0` | `0x39c` | `3`  | `0x603300` |
| `0x603300` | `0x2b3` | `4`  | `0x603310` |
| `0x603310` | `0x1dd` | `5`  | `0x603320` |
| `0x603320` | `0x1bb` | `6`  | `0x000000` |
| `0x6032d0` | `0x14c` | `1`  | `0x6032e0` |
| `0x6032e0` | `0x0a8` | `2`  | `0x6032f0` |

Column `id` gives the needed `a[]` used to build `b[]`, see [Array of Nodes](#Array-of-Nodes). So, Line 6 should begin with

```
4 3 2 1 6 5
```

The rest of the code release the frame and recover callee saved registers:


```nasm
0x4011f7 <+259>:   add    $0x50,%rsp
0x4011fb <+263>:   pop    %rbx
0x4011fc <+264>:   pop    %rbp
0x4011fd <+265>:   pop    %r12
0x4011ff <+267>:   pop    %r13
0x401201 <+269>:   pop    %r14
0x401203 <+271>:   retq
```

