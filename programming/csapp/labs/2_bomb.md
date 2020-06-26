# Bomb Lab

## Safe Functions

### `strings_length`

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

## Dangerous Functions

### `explode_bomb`

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

