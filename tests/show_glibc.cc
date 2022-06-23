// Show system GLIBC version
// Compile with
// g++ show_glibc.cc -o show_glibc.exe

#include <gnu/libc-version.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    // Method 1: use macro
    printf("%d.%d\n", __GLIBC__, __GLIBC_MINOR__);

    // Method 2: use gnu_get_libc_version 
    puts(gnu_get_libc_version());

    // Method 3: use confstr function
    char version[30] = {0};
    confstr(_CS_GNU_LIBC_VERSION, version, 30);
    puts(version);

    return 0;
}
