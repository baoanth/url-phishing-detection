#!/usr/bin/env python3
import time
from train_phishing_detector import legit_domains, domain_length_table, min_levenshtein_distance
import difflib

# Test domain
test_domain = 'googel.com'  # Typo của google.com
print(f'Testing domain: {test_domain} (length: {len(test_domain)})')
print(f'Total legit domains: {len(legit_domains)}')

# Phân tích phân bố độ dài
length_stats = {}
for domain in legit_domains:
    l = len(domain)
    length_stats[l] = length_stats.get(l, 0) + 1

print(f'Length distribution: {sorted(length_stats.items())}')

# Tính số domain trong khoảng [len-2, len+2] cho test_domain
target_len = len(test_domain)
count_in_range = 0
for check_len in range(max(1, target_len - 2), target_len + 3):
    if check_len in domain_length_table:
        count_in_range += len(domain_length_table[check_len])

print(f'Domains in length range [{target_len-2}, {target_len+2}]: {count_in_range}')
print(f'Top 100 vs Length filter: 100 vs {count_in_range}')

# Benchmark cả hai phương pháp
def old_method(domain, legit_domains):
    """Phương pháp cũ: top 100"""
    max_ratio = 0
    for legit in legit_domains[:100]:
        ratio = difflib.SequenceMatcher(None, domain, legit).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
        if ratio > 0.9:
            break
    return 1 - max_ratio

# So sánh thời gian thực thi
print("\n=== BENCHMARK ===")

# Test old method
start_time = time.time()
for _ in range(100):
    result_old = old_method(test_domain, legit_domains)
old_time = time.time() - start_time

# Test new method  
start_time = time.time()
for _ in range(100):
    result_new = min_levenshtein_distance(test_domain, domain_length_table)
new_time = time.time() - start_time

print(f"Old method (top 100): {old_time:.4f}s, Result: {result_old:.4f}")
print(f"New method (length filter): {new_time:.4f}s, Result: {result_new:.4f}")
print(f"Speedup: {old_time/new_time:.2f}x")

# Test với nhiều domain khác nhau
test_domains = ['googel.com', 'yahooo.com', 'faecbook.com', 'amason.com', 'twiter.com']

print(f"\n=== TEST WITH MULTIPLE DOMAINS ===")
for domain in test_domains:
    target_len = len(domain)
    count_in_range = 0
    for check_len in range(max(1, target_len - 2), target_len + 3):
        if check_len in domain_length_table:
            count_in_range += len(domain_length_table[check_len])
    print(f"{domain} (len={target_len}): filter={count_in_range} vs top100=100")
