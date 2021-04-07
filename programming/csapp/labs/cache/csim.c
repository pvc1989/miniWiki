#include "cachelab.h"

#include <assert.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LENGTH 128
char line[MAX_LENGTH];

bool verbose = false;

typedef struct {
  long tag;
  int valid, rank;
} line_t;

typedef struct {
  line_t** sets;
  long set_mask, tag_mask;
  int n_sets, n_lines, block_size;
  int n_hits, n_misses, n_evictions;
} cache_t;

void usage(char* csim) {
  printf("%s: Missing required command line argument\n", csim);
  printf("Usage: %s [-hv] -s <num> -E <num> -b <num> -t <file>\n", csim);
  printf("Options:\n");
  printf("  -h         Print this help message.\n");
  printf("  -v         Optional verbose flag.\n");
  printf("  -s <num>   Number of set index bits.\n");
  printf("  -E <num>   Number of lines per set.\n");
  printf("  -b <num>   Number of block offset bits.\n");
  printf("  -t <file>  Trace file.\n");
  printf("\n");
  printf("Examples:\n");
  printf("  linux>  %s -s 4 -E 1 -b 4 -t traces/yi.trace\n", csim);
  printf("  linux>  %s -v -s 8 -E 2 -b 4 -t traces/yi.trace\n", csim);
}

long hex_to_long(char* str) {
  long x = 0;
  char c;
  while ((c = *str++) != ',') {
    x *= 16l;
    if ('0' <= c && c <= '9') {
      x += (c - '0');
    } else if ('a' <= c && c <= 'f') {
      x += 10l + (c - 'a');
    } else if ('A' <= c && c <= 'F') {
      x += 10l + (c - 'A');
    } else {
      assert(false);
    }
  }
  return x;
}

void grow_rank(line_t *line_0, long n_lines, int rank_evicted) {
  assert(rank_evicted < n_lines);
  for (long i_line = 0; i_line != n_lines; ++i_line) {
    line_t *line_i = line_0 + i_line;
    if (line_i->valid && line_i->rank < rank_evicted) {
      line_i->rank += 1;
    }
  }
}

void touch(cache_t *cache, long address) {
  long tag = cache->tag_mask & address;
  long i_set = (cache->set_mask & address) / cache->block_size;
  long i_line_hit = -1, i_line_invalid = -1, i_line_evicted = -1;
  int rank_evicted = -1;
  line_t *line_0 = (void *) cache->sets + sizeof(line_t) * cache->n_lines * i_set;
  line_t *line_i;
  for (long i_line = 0; i_line != cache->n_lines; ++i_line) {
    line_i = line_0 + i_line;
    if (verbose) printf("cache[i_set = 0x%lx][i_line = 0x%lx].tag = 0x%lx, tag = 0x%lx\n", i_set, i_line, line_i->tag, tag);
    if (line_i->valid) {
      if (line_i->tag == tag) {
        rank_evicted = line_i->rank;
        i_line_hit = i_line;
        break;
      } else if (rank_evicted < line_i->rank) {
        rank_evicted = line_i->rank;
        i_line_evicted = i_line;
      }
    } else {
      i_line_invalid = i_line;
    }
  }
  if (i_line_hit != -1) {
    cache->n_hits += 1;
    if (verbose) printf("cache[i_set = 0x%lx][i_line = 0x%lx] is hit.\n", i_set, i_line_hit);
  } else {
    cache->n_misses += 1;
    if (verbose) printf("cache[i_set = 0x%lx] is miss.\n", i_set);
    if (i_line_invalid != -1) {
      line_i = line_0 + i_line_invalid;
      line_i->valid = true;
      rank_evicted = cache->n_lines - 1;
    } else {
      cache->n_evictions += 1;
      if (verbose) printf("cache[i_set = 0x%lx][i_line = 0x%lx] is evicted.\n", i_set, i_line_evicted);
      line_i = line_0 + i_line_evicted;
      assert(line_i->valid);
    }
    line_i->tag = tag;
  }
  grow_rank(line_0, cache->n_lines, rank_evicted);
  line_i->rank = 0;
  if (verbose) printf("cache[i_set = 0x%lx][i_line = 0x%lx] is used.\n", i_set, line_i - line_0);
}

int main(int argc, char* argv[]) {
  int m = 64, b, s, i;
  char *file_name; FILE *file_ptr;
  cache_t cache; cache.n_hits = cache.n_misses = cache.n_evictions = 0;

  /* parse command line options */
  int opt;
  while ((opt = getopt(argc, argv, "hvs:E:b:t:")) != -1) {
    switch (opt) {
    case 's':
      s = atoi(optarg);
      cache.n_sets = 1 << s;
      break;
    case 'E':
      cache.n_lines = atoi(optarg);
      break;
    case 'b':
      b = atoi(optarg);
      cache.block_size = 1 << b;
      break;
    case 't':
      file_name = optarg;
      if ((file_ptr = fopen(file_name, "r")) == NULL) {
        printf("Cannot open `%s`.\n", file_name);
        return 1;
      }
      break;
    case 'v':
      verbose = true;
      break;
    case '?':
    case 'h':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  if (verbose) {
    printf("#sets = %d\n", cache.n_sets);
    printf("#lines / set = %d\n", cache.n_lines);
    printf("#bytes / block = %d\n", cache.block_size);
    printf("file = %s\n", file_name);
    printf("\n");
  }

  cache.set_mask = cache.tag_mask = 0;
  for (i = b; i < b + s; ++i)
    cache.set_mask |= 1L << i;
  for (i = b + s; i < m; ++i)
    cache.tag_mask |= 1L << i;
  if (verbose) {
    printf("set_mask = 0x%.16lx\n", cache.set_mask);
    printf("tag_mask = 0x%.16lx\n", cache.tag_mask);
    printf("\n");
  }

  int n_bytes = sizeof(line_t) * cache.n_sets * cache.n_lines;
  cache.sets = (line_t**) malloc(n_bytes);
  if (cache.sets == NULL) {
    printf("Cannot allocate %d bytes.\n", n_bytes);
    return 1;
  }
  memset(cache.sets, 0, n_bytes);

  while (fgets(line, MAX_LENGTH, file_ptr)) {
    if (verbose) printf("%s", line);
    long address = hex_to_long(line + 3);
    switch (line[1]) {
    case 'L':
      if (verbose) printf(" Load from 0x%lx.\n", address);
      touch(&cache, address);
      break;
    case 'M':
      if (verbose) printf(" Modify on 0x%lx.\n", address);
      touch(&cache, address);
      touch(&cache, address);
      break;
    case 'S':
      if (verbose) printf(" Save into 0x%lx.\n", address);
      touch(&cache, address);
      break;
    default:
      if (verbose) printf("Unknown operation.\n");
      break;
    }
  }

  printSummary(cache.n_hits, cache.n_misses, cache.n_evictions);

  /* clean up */
  free(cache.sets);
  fclose(file_ptr);
  return 0;
}
