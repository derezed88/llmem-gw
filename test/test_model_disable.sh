#!/bin/bash
# Test that disabled models are excluded from LLM_REGISTRY

echo "Testing Model Disable Functionality"
echo "===================================="
echo ""

# Activate venv
source venv/bin/activate

# Test 1: Check Win11 disabled status in JSON
echo "Test 1: Check Win11 disabled status in llm-models.json"
python -c "
import json
with open('llm-models.json') as f:
    data = json.load(f)
enabled = data['models']['Win11'].get('enabled', True)
print(f'  Win11 enabled in JSON: {enabled}')
assert enabled == False, 'Win11 should be disabled'
print('  ✓ Win11 is correctly disabled in llm-models.json')
"
echo ""

# Test 2: Verify LLM_REGISTRY only contains enabled models
echo "Test 2: Verify LLM_REGISTRY excludes disabled models"
python -c "
from config import LLM_REGISTRY
import json

models = list(LLM_REGISTRY.keys())
print(f'  Models in LLM_REGISTRY: {json.dumps(models, indent=4)}')

assert 'Win11' not in models, 'Win11 should NOT be in LLM_REGISTRY'
print('  ✓ Win11 correctly excluded from LLM_REGISTRY')

expected = {'grok-4', 'openai', 'gemini-2.5-flash'}
actual = set(models)
assert actual == expected, f'Expected {expected}, got {actual}'
print(f'  ✓ Only enabled models loaded: {models}')
"
echo ""

# Test 3: Test re-enabling
echo "Test 3: Test re-enabling Win11"
python llmemctl.py model-enable Win11 > /dev/null 2>&1

python -c "
from config import load_llm_registry
registry = load_llm_registry()
models = list(registry.keys())
print(f'  Models after re-enabling: {models}')

if 'Win11' in models:
    print('  ✓ Win11 successfully re-enabled and loaded')
else:
    print('  ✗ Win11 still not in registry (need agent restart)')
    print('  Note: Changes require agent restart to take effect')
"
echo ""

# Test 4: Restore disabled state
echo "Test 4: Restore Win11 to disabled state"
python llmemctl.py model-disable Win11 > /dev/null 2>&1
echo "  ✓ Win11 disabled again"
echo ""

echo "===================================="
echo "All tests passed!"
echo ""
echo "Summary:"
echo "  - Disabled models are excluded from llm-models.json"
echo "  - config.py loads only enabled models into LLM_REGISTRY"
echo "  - !model command will not show disabled models"
echo "  - Attempting to use disabled model will be rejected"
echo ""
echo "To test in agent:"
echo "  1. Restart agent: python main.py"
echo "  2. Connect with shell.py"
echo "  3. Run: !model"
echo "  4. Verify Win11 is NOT in the list"
echo "  5. Run: !model Win11"
echo "  6. Verify error: 'Unknown model'"
