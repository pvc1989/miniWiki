# Bomb Lab

## Dangerous Functions

### `explode_bomb`

```c
void explode_bomb();
```

👉 Set a breakpoint at the head of this function to avoid explosion.

## Phase 1

### `string_length`

```c
int string_length(char* s);
```

### `strings_not_equal`

```assembly
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

### `phase_1`

```assembly
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

## Phase 2

### `read_six_numbers`

```assembly
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

### `phase_2`

```assembly
Dump of assembler code for function phase_2:
   0x400efc <+0>:     push   %rbp
   0x400efd <+1>:     push   %rbx
   0x400efe <+2>:     sub    $0x28,%rsp  # allocate local array
   0x400f02 <+6>:     mov    %rsp,%rsi   # head of the array
   0x400f05 <+9>:     callq  0x40145c <read_six_numbers>
                      # 用 `x/6wd $rsp` 可查看这 6 个数
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

