---
title: Bomb Lab
---

# Resources
- [Linux/x86-64 binary bomb](http://csapp.cs.cmu.edu/3e/bomb.tar) for self-study
- [Recitation 4: Bomb Lab](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=0ed08c72-f0f1-4982-aebc-92d3db7af9fd)

# Hints

## One Solution

There are more than one possible solutions, one of which is

```
Border relations with Canada have never been better.
1 2 4 8 16 32
7 327
0 0
IONEFG
4 3 2 1 6 5
```

Give them to the `bomb`, you will get these output:

```
Welcome to my fiendish little bomb. You have 6 phases with
which to blow yourself up. Have a nice day!
Phase 1 defused. How about the next one?
That's number 2.  Keep going!
Halfway there!
So you got that one.  Try this one.
Good work!  On to the next...
Congratulations! You've defused the bomb!
```

## Dangerous Functions

The following function and those who call it are dangerous:

```c
void explode_bomb();
```

👉 Set a breakpoint at the entry of this function to avoid explosion.

```shell
(gdb/lldb) b explode_bomb
```

# Phase 1

## `string_length()`

```nasm
0x40131b <+0>:     cmp    BYTE PTR [rdi],0x0
0x40131e <+3>:     je     0x401332 <string_length+23>
0x401320 <+5>:     mov    rdx,rdi
0x401323 <+8>:     add    rdx,0x1
0x401327 <+12>:    mov    eax,edx
0x401329 <+14>:    sub    eax,edi
0x40132b <+16>:    cmp    BYTE PTR [rdx],0x0
0x40132e <+19>:    jne    0x401323 <string_length+8>
0x401330 <+21>:    repz ret 
0x401332 <+23>:    mov    eax,0x0
0x401337 <+28>:    ret
```

```c
int string_length(char* s);
```

## `strings_not_equal()`

```nasm
0x401338 <+0>:     push   r12
0x40133a <+2>:     push   rbp
0x40133b <+3>:     push   rbx
0x40133c <+4>:     mov    rbx,rdi
0x40133f <+7>:     mov    rbp,rsi
0x401342 <+10>:    call   0x40131b <string_length>  ; l1 = string_length(s1);
0x401347 <+15>:    mov    r12d,eax
0x40134a <+18>:    mov    rdi,rbp
0x40134d <+21>:    call   0x40131b <string_length>  ; l2 = string_length(s2);
0x401352 <+26>:    mov    edx,0x1
0x401357 <+31>:    cmp    r12d,eax  ; if (l1 == l2)
0x40135a <+34>:    jne    0x40139b <strings_not_equal+99>  ; else
0x40135c <+36>:    movzx  eax,BYTE PTR [rbx]  ; R[al] = s1[0]
0x40135f <+39>:    test   al,al  ; if (s1[0] == '\0')
0x401361 <+41>:    je     0x401388 <strings_not_equal+80>  ; then
; else, i.e. (s1[0] != '\0')
0x401363 <+43>:    cmp    al,BYTE PTR [rbp+0x0]  ; if (*s1 == *s2)
0x401366 <+46>:    je     0x401372 <strings_not_equal+58>  ; then
0x401368 <+48>:    jmp    0x40138f <strings_not_equal+87>  ; else
0x40136a <+50>:    cmp    al,BYTE PTR [rbp+0x0]  ; if (*s1 == *s2)
0x40136d <+53>:    nop    DWORD PTR [rax]  ; no operation
0x401370 <+56>:    jne    0x401396 <strings_not_equal+94>  ; else
0x401372 <+58>:    add    rbx,0x1  ; ++s1
0x401376 <+62>:    add    rbp,0x1  ; ++s2
0x40137a <+66>:    movzx  eax,BYTE PTR [rbx]  ; R[al] = *s1
0x40137d <+69>:    test   al,al  ; if (*s1 == '\0')
0x40137f <+71>:    jne    0x40136a <strings_not_equal+50>  ; else
; return 0;
0x401381 <+73>:    mov    edx,0x0  ; *s1 == '\0'
0x401386 <+78>:    jmp    0x40139b <strings_not_equal+99>
0x401388 <+80>:    mov    edx,0x0  ; s1[0] == '\0'
0x40138d <+85>:    jmp    0x40139b <strings_not_equal+99>
; return 1;
0x40138f <+87>:    mov    edx,0x1  ; *s1 != *s2
0x401394 <+92>:    jmp    0x40139b <strings_not_equal+99>
0x401396 <+94>:    mov    edx,0x1  ; temp = 1;
0x40139b <+99>:    mov    eax,edx  ; result = temp;
0x40139d <+101>:   pop    rbx
0x40139e <+102>:   pop    rbp
0x40139f <+103>:   pop    r12
0x4013a1 <+105>:   ret
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

## `phase_1()`

```nasm
0x400ee0 <+0>:     sub    rsp,0x8
0x400ee4 <+4>:     mov    esi,0x402400
0x400ee9 <+9>:     call   0x401338 <strings_not_equal>
0x400eee <+14>:    test   eax,eax
0x400ef0 <+16>:    je     0x400ef7 <phase_1+23>
0x400ef2 <+18>:    call   0x40143a <explode_bomb>
0x400ef7 <+23>:    add    rsp,0x8
0x400efb <+27>:    ret
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

# Phase 2

## `read_six_numbers()`

```nasm
0x40145c <+0>:     sub    rsp,0x18
0x401460 <+4>:     mov    rdx,rsi
0x401463 <+7>:     lea    rcx,[rsi+0x4]
0x401467 <+11>:    lea    rax,[rsi+0x14]
0x40146b <+15>:    mov    QWORD PTR [rsp+0x8],rax
0x401470 <+20>:    lea    rax,[rsi+0x10]
0x401474 <+24>:    mov    QWORD PTR [rsp],rax
0x401478 <+28>:    lea    r9,[rsi+0xc]
0x40147c <+32>:    lea    r8,[rsi+0x8]
0x401480 <+36>:    mov    esi,0x4025c3   ; "%d %d %d %d %d %d"
0x401485 <+41>:    mov    eax,0x0        ; n = 0
0x40148a <+46>:    call   0x400bf0 <__isoc99_sscanf@plt>
0x40148f <+51>:    cmp    eax,0x5                         ; n -  5
0x401492 <+54>:    jg     0x401499 <read_six_numbers+61>  ; n >  5
0x401494 <+56>:    call   0x40143a <explode_bomb>         ; n <= 5 
0x401499 <+61>:    add    rsp,0x18
0x40149d <+65>:    ret
```

```c
int read_six_numbers(char* s, int a[]);
```

## `phase_2()`

```nasm
0x400efc <+0>:     push   rbp
0x400efd <+1>:     push   rbx
0x400efe <+2>:     sub    rsp,0x28  ; allocate local array
0x400f02 <+6>:     mov    rsi,rsp   ; head of the array
0x400f05 <+9>:     call   0x40145c <read_six_numbers>
; use `x/6wd $rsp` to examine these numbers
0x400f0a <+14>:    cmp    DWORD PTR [rsp],0x1      ; a[0] -  1
0x400f0e <+18>:    je     0x400f30 <phase_2+52>    ; a[0] == 1
0x400f10 <+20>:    call   0x40143a <explode_bomb>  ; a[0] != 1
0x400f15 <+25>:    jmp    0x400f30 <phase_2+52>
0x400f17 <+27>:    mov    eax,DWORD PTR [rbx-0x4]  ; R[eax] = a[i-1]
0x400f1a <+30>:    add    eax,eax                  ; R[eax] = a[i-1] + a[i-1]
0x400f1c <+32>:    cmp    DWORD PTR [rbx],eax      ; a[i] -  2*a[i-1]
0x400f1e <+34>:    je     0x400f25 <phase_2+41>    ; a[i] == 2*a[i-1]
0x400f20 <+36>:    call   0x40143a <explode_bomb>  ; a[i] != 2*a[i-1]
0x400f25 <+41>:    add    rbx,0x4  ; ++i
0x400f29 <+45>:    cmp    rbx,rbp                ; &a[i] -  &a[5]
0x400f2c <+48>:    jne    0x400f17 <phase_2+27>  ; &a[i] != &a[5]
0x400f2e <+50>:    jmp    0x400f3c <phase_2+64>  ; &a[i] == &a[5]
0x400f30 <+52>:    lea    rbx,[rsp+0x4]   ; R[rbx] = &a[1]
0x400f35 <+57>:    lea    rbp,[rsp+0x18]  ; R[rbp] = &a[5]
0x400f3a <+62>:    jmp    0x400f17 <phase_2+27>
0x400f3c <+64>:    add    rsp,0x28
0x400f40 <+68>:    pop    rbx
0x400f41 <+69>:    pop    rbp
0x400f42 <+70>:    ret
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

# Phase 3

## `phase_3()`

```nasm
0x400f43 <+0>:     sub    rsp,0x18
0x400f47 <+4>:     lea    rcx,[rsp+0xc]  ; int y
0x400f4c <+9>:     lea    rdx,[rsp+0x8]  ; int x
0x400f51 <+14>:    mov    esi,0x4025cf   ; "%d %d"
0x400f56 <+19>:    mov    eax,0x0
0x400f5b <+24>:    call   0x400bf0 <__isoc99_sscanf@plt>
0x400f60 <+29>:    cmp    eax,0x1                  ; n -  1
0x400f63 <+32>:    jg     0x400f6a <phase_3+39>    ; n >  1
0x400f65 <+34>:    call   0x40143a <explode_bomb>  ; n <= 1
0x400f6a <+39>:    cmp    DWORD PTR [rsp+0x8],0x7  ; x -  7
0x400f6f <+44>:    ja     0x400fad <phase_3+106>   ; x >  7
0x400f71 <+46>:    mov    eax,DWORD PTR [rsp+0x8]  ; x <= 7
; enter the switch
0x400f75 <+50>:    jmp    QWORD PTR [rax*8+0x402470]  ; switch (x)
; use `x/8gx 0x402470` to examine the table, which gives
;   0x402470: 0x0000000000400f7c 0x0000000000400fb9
;   0x402480: 0x0000000000400f83 0x0000000000400f8a
;   0x402490: 0x0000000000400f91 0x0000000000400f98
;   0x4024a0: 0x0000000000400f9f 0x0000000000400fa6
; thus we have
; case 0:
0x400f7c <+57>:    mov    eax,0xcf
0x400f81 <+62>:    jmp    0x400fbe <phase_3+123>
; case 2:
0x400f83 <+64>:    mov    eax,0x2c3
0x400f88 <+69>:    jmp    0x400fbe <phase_3+123>
; case 3:
0x400f8a <+71>:    mov    eax,0x100
0x400f8f <+76>:    jmp    0x400fbe <phase_3+123>
; case 4:
0x400f91 <+78>:    mov    eax,0x185
0x400f96 <+83>:    jmp    0x400fbe <phase_3+123>
; case 5:
0x400f98 <+85>:    mov    eax,0xce
0x400f9d <+90>:    jmp    0x400fbe <phase_3+123>
; case 6:
0x400f9f <+92>:    mov    eax,0x2aa
0x400fa4 <+97>:    jmp    0x400fbe <phase_3+123>
; case 7:
0x400fa6 <+99>:    mov    eax,0x147
0x400fab <+104>:   jmp    0x400fbe <phase_3+123>
; default:
0x400fad <+106>:   call   0x40143a <explode_bomb>  ; (unsigned) x > 7
0x400fb2 <+111>:   mov    eax,0x0
0x400fb7 <+116>:   jmp    0x400fbe <phase_3+123>
; case 1:
0x400fb9 <+118>:   mov    eax,0x137
; leave the switch
0x400fbe <+123>:   cmp    eax,DWORD PTR [rsp+0xc]  ; t -  y
0x400fc2 <+127>:   je     0x400fc9 <phase_3+134>   ; t == y
0x400fc4 <+129>:   call   0x40143a <explode_bomb>  ; t != y
0x400fc9 <+134>:   add    rsp,0x18
0x400fcd <+138>:   ret
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
    t = 0x0cf;  /* 207 */
    break;
  case 1:
    t = 0x137;  /* 311 */
    break;
  case 2:
    t = 0x2c3;  /* 707 */
    break;
  case 3:
    t = 0x100;  /* 256 */
    break;
  case 4:
    t = 0x185;  /* 389 */
    break;
  case 5:
    t = 0x0ce;  /* 206*/
    break;
  case 6:
    t = 0x2aa;  /* 682 */
    break;
  case 7:
    t = 0x147;  /* 327 */
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
3 256
4 389
5 206
6 682
7 327
```

# Phase 4

## `func4()`

```nasm
0x400fce <+0>:     sub    rsp,0x8
0x400fd2 <+4>:     mov    eax,edx   ; t = c
0x400fd4 <+6>:     sub    eax,esi   ; t = c - b
0x400fd6 <+8>:     mov    ecx,eax   ; s = t
0x400fd8 <+10>:    shr    ecx,0x1f  ; s /= (1<<31)
0x400fdb <+13>:    add    eax,ecx   ; t += s
0x400fdd <+15>:    sar    eax,1     ; t /= 2
0x400fdf <+17>:    lea    ecx,[rax+rsi*1]  ; s = t + b
0x400fe2 <+20>:    cmp    ecx,edi   ; s - a
0x400fe4 <+22>:    jle    0x400ff2 <func4+36>
; s > a
0x400fe6 <+24>:    lea    edx,[rcx-0x1]
0x400fe9 <+27>:    call   0x400fce <func4>  ; t = func4(a, b, s-1)
0x400fee <+32>:    add    eax,eax           ; t += t
0x400ff0 <+34>:    jmp    0x401007 <func4+57>
; s <= a
0x400ff2 <+36>:    mov    eax,0x0  ; t = 0
0x400ff7 <+41>:    cmp    ecx,edi  ; s - a
0x400ff9 <+43>:    jge    0x401007 <func4+57>
0x400ffb <+45>:    lea    esi,[rcx+0x1]
0x400ffe <+48>:    call   0x400fce <func4>     ; t = func4(a, s+1, c)
0x401003 <+53>:    lea    eax,[rax+rax*1+0x1]  ; t = 1 + 2*t
; return t
0x401007 <+57>:    add    rsp,0x8
0x40100b <+61>:    ret
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

## `phase_4()`

```nasm
0x40100c <+0>:     sub    rsp,0x18
0x401010 <+4>:     lea    rcx,[rsp+0xc]
0x401015 <+9>:     lea    rdx,[rsp+0x8]
0x40101a <+14>:    mov    esi,0x4025cf  ; "%d %d"
0x40101f <+19>:    mov    eax,0x0
0x401024 <+24>:    call   0x400bf0 <__isoc99_sscanf@plt>
0x401029 <+29>:    cmp    eax,0x2                ; n -  2
0x40102c <+32>:    jne    0x401035 <phase_4+41>  ; n != 2
0x40102e <+34>:    cmp    DWORD PTR [rsp+0x8],0xe  ; x - 0xe
0x401033 <+39>:    jbe    0x40103a <phase_4+46>
0x401035 <+41>:    call   0x40143a <explode_bomb>  ; x > 0xe
0x40103a <+46>:    mov    edx,0xe
0x40103f <+51>:    mov    esi,0x0
0x401044 <+56>:    mov    edi,DWORD PTR [rsp+0x8]
0x401048 <+60>:    call   0x400fce <func4>  ; z = func4(x, 0, 14)
0x40104d <+65>:    test   eax,eax  ; z & z
0x40104f <+67>:    jne    0x401058 <phase_4+76>    ; z != 0
0x401051 <+69>:    cmp    DWORD PTR [rsp+0xc],0x0  ; y -  0
0x401056 <+74>:    je     0x40105d <phase_4+81>    ; y == 0
0x401058 <+76>:    call   0x40143a <explode_bomb>  ; y != 0
0x40105d <+81>:    add    rsp,0x18
0x401061 <+85>:    ret
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

# Phase 5

## `phase_5()`

```nasm
0x401062 <+0>:     push   rbx
0x401063 <+1>:     sub    rsp,0x20
0x401067 <+5>:     mov    rbx,rdi
0x40106a <+8>:     mov    rax,QWORD PTR fs:0x28  ; store canary
0x401073 <+17>:    mov    QWORD PTR [rsp+0x18],rax
0x401078 <+22>:    xor    eax,eax
0x40107a <+24>:    call   0x40131b <string_length>
0x40107f <+29>:    cmp    eax,0x6
0x401082 <+32>:    je     0x4010d2 <phase_5+112>  ; wanted
0x401084 <+34>:    call   0x40143a <explode_bomb>
0x401089 <+39>:    jmp    0x4010d2 <phase_5+112>
; enter the loop
0x40108b <+41>:    movzx  ecx,BYTE PTR [rbx+rax*1]  ; int R[ecx] = a[i]
0x40108f <+45>:    mov    BYTE PTR [rsp],cl    ; char M[R[rsp]] = a[i]
0x401092 <+48>:    mov    rdx,QWORD PTR [rsp]  ; long R[rdx] = M[R[rsp]]
0x401096 <+52>:    and    edx,0xf    ; a[i] is the least significant byte
0x401099 <+55>:    movzx  edx,BYTE PTR [rdx+0x4024b0]  ; s = M[0x4024b0 + s]
0x4010a0 <+62>:    mov    BYTE PTR [rsp+rax*1+0x10],dl  ; a[i] = s
0x4010a4 <+66>:    add    rax,0x1  ; ++i
0x4010a8 <+70>:    cmp    rax,0x6  ; i - 6
0x4010ac <+74>:    jne    0x40108b <phase_5+41>  ; while (i != 0)
; leave the loop
0x4010ae <+76>:    mov    BYTE PTR [rsp+0x16],0x0  ; a[6] == '\0'
0x4010b3 <+81>:    mov    esi,0x40245e    ; "flyers" as the 2nd arg
0x4010b8 <+86>:    lea    rdi,[rsp+0x10]  ;   &a[0]  as the 1st arg
0x4010bd <+91>:    call   0x401338 <strings_not_equal>
0x4010c2 <+96>:    test   eax,eax
0x4010c4 <+98>:    je     0x4010d9 <phase_5+119>  ; wanted
0x4010c6 <+100>:   call   0x40143a <explode_bomb>
0x4010cb <+105>:   nop    DWORD PTR [rax+rax*1+0x0]
0x4010d0 <+110>:   jmp    0x4010d9 <phase_5+119>
0x4010d2 <+112>:   mov    eax,0x0
0x4010d7 <+117>:   jmp    0x40108b <phase_5+41>
0x4010d9 <+119>:   mov    rax,QWORD PTR [rsp+0x18]  ; the stored canary
0x4010de <+124>:   xor    rax,QWORD PTR fs:0x28
0x4010e7 <+133>:   je     0x4010ee <phase_5+140>  ; no overflow
0x4010e9 <+135>:   call   0x400b30 <__stack_chk_fail@plt>
0x4010ee <+140>:   add    rsp,0x20  ; clean up
0x4010f2 <+144>:   pop    rbx
0x4010f3 <+145>:   ret
```

## Key Steps

1. From `<+24>` to `<+34>`, we know ***Line 5 should be a 6-`char` string (excluding the `'\0'` at the end)***.

1. There is a 6-step loop between `<+41>` and `<+74>`, which does not call `explode_bomb()`. So, it's safe to set a breakpoint after it, and let the process keep running until `<+76>` is hit.

1. From `<+76>` to `<+100>`, we know ***the target string at `0x40245e` is `flyers`***, which is indeed 6-`char` long.

1. Now, it's time to examine the loop in detail. We are working on a *little-endian* system, so `a[i]` resides in the least-significant byte, and the the net effect of the 3 instructions beginning at `<+81>` is
   ```c
   R[rdx] = ZeroExtend(a[i] % 0x10);
   ```
   So, the C code for the loop might be
   ```c
   void phase_5(char* input) {
     if (string_length(input) != 6) explode_bomb();
     for (char* curr = a; curr != a+6; ++curr) {
       int t = *curr;    /* ecx */
       long s = t % 16;  /* rdx */
       s = *(int *)(0x4024b0 + s);
       *curr = (char) s;
     }
   }
   ```
   
1. The 16 bytes beginning at `0x4024b0`, denoted as `char b[16]`, are
   ```shell
   (lldb) x/16bd 0x4024b0
   0x4024b0: 109
   0x4024b1: 97
   0x4024b2: 100
   0x4024b3: 117
   0x4024b4: 105
   0x4024b5: 101
   0x4024b6: 114
   0x4024b7: 115
   0x4024b8: 110
   0x4024b9: 102
   0x4024ba: 111
   0x4024bb: 116
   0x4024bc: 118
   0x4024bd: 98
   0x4024be: 121
   0x4024bf: 108
   ```
   while the 6 bytes in `flyers`, beginning at `0x40245e`, denoted as `char c[6]`, are
   ```shell
   (lldb) x/6bd 0x40245e
   0x40245e: 102
   0x40245f: 108
   0x402460: 121
   0x402461: 101
   0x402462: 114
   0x402463: 115
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

# Phase 6

## `read_six_numbers()`

```nasm
0x4010f4 <+0>:     push   r14
0x4010f6 <+2>:     push   r13
0x4010f8 <+4>:     push   r12
0x4010fa <+6>:     push   rbp
0x4010fb <+7>:     push   rbx
0x4010fc <+8>:     sub    rsp,0x50  ; allocate array
0x401100 <+12>:    mov    r13,rsp
0x401103 <+15>:    mov    rsi,rsp
0x401106 <+18>:    call   0x40145c <read_six_numbers>
0x40110b <+23>:    mov    r14,rsp  ; save &a[0] for later use
```
`read_six_numbers()` implies ***Line 6 must begin with 6 integers***. So, the first 2 lines in the C code looks like

```c
int a[6];
read_six_numbers(input, a);  /* <+18> */
```

## `1, 2, 3, 4, 5, 6`

```nasm
; initialize the loop
0x40110e <+26>:    mov    r12d,0x0  ; int i = 0
; enter the loop
0x401114 <+32>:    mov    rbp,r13  ; R[rbp] == &a[i]
0x401117 <+35>:    mov    eax,DWORD PTR [r13+0x0]  ; unsigned int t = a[i]
0x40111b <+39>:    sub    eax,0x1  ; t -= 1
0x40111e <+42>:    cmp    eax,0x5                  ;          t -  5
0x401121 <+45>:    jbe    0x401128 <phase_6+52>    ; unsigned t <= 5
0x401123 <+47>:    call   0x40143a <explode_bomb>  ;          t >  5
0x401128 <+52>:    add    r12d,0x1  ; ++i
0x40112c <+56>:    cmp    r12d,0x6               ; i -  6
0x401130 <+60>:    je     0x401153 <phase_6+95>  ; i == 6
; enter the inner loop
0x401132 <+62>:    mov    ebx,r12d  ; int j = i
0x401135 <+65>:    movsxd rax,ebx   ; R[rax] = j
0x401138 <+68>:    mov    eax,DWORD PTR [rsp+rax*4]  ; R[eax] = a[j]
0x40113b <+71>:    cmp    DWORD PTR [rbp+0x0],eax    ; a[i-1] -  a[j]
0x40113e <+74>:    jne    0x401145 <phase_6+81>      ; a[i-1] != a[j]
0x401140 <+76>:    call   0x40143a <explode_bomb>    ; a[i-1] == a[j]
0x401145 <+81>:    add    ebx,0x1  ; ++j
0x401148 <+84>:    cmp    ebx,0x5                ; j -  5
0x40114b <+87>:    jle    0x401135 <phase_6+65>  ; j <= 5
; leave the inner loop
0x40114d <+89>:    add    r13,0x4  ; a += 1
0x401151 <+93>:    jmp    0x401114 <phase_6+32>
; leave the loop
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

```shell
(lldb) x/6wd $rsp
0x7fffffffe4c0: 1
0x7fffffffe4c4: 2
0x7fffffffe4c8: 3
0x7fffffffe4cc: 4
0x7fffffffe4d0: 5
0x7fffffffe4d4: 6
```

## `a[i] = 7 - a[i]`

```nasm
; initialize the loop
0x401153 <+95>:    lea    rsi,[rsp+0x18]  ; int* a_n = &a[6]
0x401158 <+100>:   mov    rax,r14         ; int* a_i = &a[i]
0x40115b <+103>:   mov    ecx,0x7  ; t = 7
; enter the loop
0x401160 <+108>:   mov    edx,ecx              ; s = t
0x401162 <+110>:   sub    edx,DWORD PTR [rax]  ; s = t - a[i]
0x401164 <+112>:   mov    DWORD PTR [rax],edx  ; a[i] = 7 - a[i] 👈
0x401166 <+114>:   add    rax,0x4  ; a_i += 1
0x40116a <+118>:   cmp    rax,rsi  ; a_i - a_n
0x40116d <+121>:   jne    0x401160 <phase_6+108>  ; a_i != a_n
; leave the loop
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

## Array of Nodes

```nasm
; initialize the loop
0x40116f <+123>:   mov    esi,0x0  ; void* offset = 0
; enter the loop
0x401174 <+128>:   jmp    0x401197 <phase_6+163>
; enter the inner loop
0x401176 <+130>:   mov    rdx,QWORD PTR [rdx+0x8]  ; target = *(target + 8)
0x40117a <+134>:   add    eax,0x1  ; count += 1
0x40117d <+137>:   cmp    eax,ecx  ; count - value
0x40117f <+139>:   jne    0x401176 <phase_6+130>  ; while (count != value)
0x401181 <+141>:   jmp    0x401188 <phase_6+148>
; leave the inner loop
0x401183 <+143>:   mov    edx,0x6032d0  ; if (value <= 1)
; after if-else
0x401188 <+148>:   mov    QWORD PTR [rsp+rsi*2+0x20],rdx  ; b[i] = target
0x40118d <+153>:   add    rsi,0x4   ; offset += 4
0x401191 <+157>:   cmp    rsi,0x18  ; offset - 24
0x401195 <+161>:   je     0x4011ab <phase_6+183>
; enter the loop
0x401197 <+163>:   mov    ecx,DWORD PTR [rsp+rsi*1]  ; int value = *(int*)((void*) a + offset)
0x40119a <+166>:   cmp    ecx,0x1  ; value - 1
0x40119d <+169>:   jle    0x401183 <phase_6+143>  ; if (value <= 1)
0x40119f <+171>:   mov    eax,0x1  ; if (value > 1) count = 1
0x4011a4 <+176>:   mov    edx,0x6032d0  ; void* target = 0x6032d0
0x4011a9 <+181>:   jmp    0x401176 <phase_6+130>  ; inner loop
; leave the loop
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

```shell
(lldb) x/12gx 0x6032d0
0x6032d0: 0x000000010000014c 0x00000000006032e0
0x6032e0: 0x00000002000000a8 0x00000000006032f0
0x6032f0: 0x000000030000039c 0x0000000000603300
0x603300: 0x00000004000002b3 0x0000000000603310
0x603310: 0x00000005000001dd 0x0000000000603320
0x603320: 0x00000006000001bb 0x0000000000000000
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

```shell
# stop at <+183>
(lldb) x/6wd $rsp  # int a[6]
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

## Rebuild the List
```nasm
0x4011ab <+183>:   mov    rbx,QWORD PTR [rsp+0x20]  ; T* rbx = b[0];  // head
0x4011b0 <+188>:   lea    rax,[rsp+0x28]  ; T** rax = &b[1];
0x4011b5 <+193>:   lea    rsi,[rsp+0x50]  ; T** rsi = &b[6];  // tail
0x4011ba <+198>:   mov    rcx,rbx         ; T* rcx = b[0];  // b[i-1]
; enter the loop
0x4011bd <+201>:   mov    rdx,QWORD PTR [rax]      ; T* rdx = *rax;  // b[i]
0x4011c0 <+204>:   mov    QWORD PTR [rcx+0x8],rdx   ; rcx->next = rdx;
0x4011c4 <+208>:   add    rax,0x8        ; rax += 1;  // &b[i+1]
0x4011c8 <+212>:   cmp    rax,rsi        ; rax - tail
0x4011cb <+215>:   je     0x4011d2 <phase_6+222>
0x4011cd <+217>:   mov    rcx,rdx        ; rcx = rdx;  // b[i]
0x4011d0 <+220>:   jmp    0x4011bd <phase_6+201>  ; while (rax != tail)
; leave the loop
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

## Sort by `rank`

At this point, `%rdx` holds `b[5]` and `%rbx` holds `b[0]`.

```nasm
0x4011d2 <+222>:   mov    QWORD PTR [rdx+0x8],0x0  ; b[5]->next = 0
0x4011da <+230>:   mov    ebp,0x5  ; i = 5
; enter the loop
0x4011df <+235>:   mov    rax,QWORD PTR [rbx+0x8]  ; rax = rbx->next
0x4011e3 <+239>:   mov    eax,DWORD PTR [rax]      ; int eax = *rax
0x4011e5 <+241>:   cmp    DWORD PTR [rbx],eax      ; rbx->data -  eax
0x4011e7 <+243>:   jge    0x4011ee <phase_6+250>   ; rbx->data >= eax
0x4011e9 <+245>:   call   0x40143a <explode_bomb>  ; rbx->data <  eax
0x4011ee <+250>:   mov    rbx,QWORD PTR [rbx+0x8]  ; rbx = rbx->next
0x4011f2 <+254>:   sub    ebp,0x1  ; --i
0x4011f5 <+257>:   jne    0x4011df <phase_6+235>
; leave the loop
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
```shell
(lldb) x/12gx 0x6032d0
0x6032d0: 0x000000010000014c 0x00000000006032e0
0x6032e0: 0x00000002000000a8 0x00000000006032f0
0x6032f0: 0x000000030000039c 0x0000000000603300
0x603300: 0x00000004000002b3 0x0000000000603310
0x603310: 0x00000005000001dd 0x0000000000603320
0x603320: 0x00000006000001bb 0x0000000000000000
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
0x4011f7 <+259>:   add    rsp,0x50
0x4011fb <+263>:   pop    rbx
0x4011fc <+264>:   pop    rbp
0x4011fd <+265>:   pop    r12
0x4011ff <+267>:   pop    r13
0x401201 <+269>:   pop    r14
0x401203 <+271>:   ret 
```

