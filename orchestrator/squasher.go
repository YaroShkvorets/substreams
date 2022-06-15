package orchestrator

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/streamingfast/substreams/block"

	"github.com/streamingfast/substreams/state"
)

// Squasher produces _complete_ stores, by merging backing partial stores.
type Squasher struct {
	squashables          map[string]*Squashable
	storeSaveInterval    uint64
	targetExclusiveBlock uint64

	notifier Notifier

	lock sync.Mutex
}

// NewSquasher receives stores, initializes them and fetches them from
// the existing storage. It prepares itself to receive Squash()
// requests that should correspond to what is missing for those stores
// to reach `targetExclusiveBlock`.  This is managed externally by the
// Scheduler/Strategy. Eventually, ideally, all components are
// synchronizes around the actual data: the state of storages
// present, the requests needed to fill in those stores up to the
// target block, etc..
func NewSquasher(ctx context.Context, workPlan WorkPlan, stores map[string]*state.Store, reqStartBlock uint64, notifier Notifier) (*Squasher, error) {
	squashables := map[string]*Squashable{}
	for storeName, store := range stores {
		workUnit := workPlan[storeName]

		var squashable *Squashable
		if workUnit.initialStoreFile == nil {
			squashable = NewSquashable(store.CloneStructure(store.ModuleInitialBlock), reqStartBlock, store.ModuleInitialBlock, notifier)
		} else {
			squish, err := store.LoadFrom(ctx, workUnit.initialStoreFile)
			if err != nil {
				return nil, fmt.Errorf("loading store %q: range %s: %w", store.Name, workUnit.initialStoreFile, err)
			}
			squashable = NewSquashable(squish, reqStartBlock, workUnit.initialStoreFile.ExclusiveEndBlock, notifier)
		}

		if len(workUnit.partialsMissing) == 0 {
			squashable.targetReached = true
			squashable.notifyWaiters(reqStartBlock)
		}

		squashables[store.Name] = squashable
	}

	squasher := &Squasher{
		squashables:          squashables,
		targetExclusiveBlock: reqStartBlock,
		notifier:             notifier,
	}

	return squasher, nil
}

func (s *Squasher) Squash(ctx context.Context, moduleName string, partialsRanges block.Ranges) error {
	s.lock.Lock()
	defer s.lock.Unlock()

	squashable, ok := s.squashables[moduleName]
	if !ok {
		return fmt.Errorf("module %q was not found in squashables module registry", moduleName)
	}

	return squashable.squash(ctx, partialsRanges)
}

func (s *Squasher) StoresReady() (out map[string]*state.Store, err error) {
	// FIXME(abourget): Before checking the state of all those squashables,
	// we need to make sure all those Scheduler::Callback and Squash() calls
	// have finished, and that those `merge()` operations have completed..
	// otherwise here we can't loop the squashables and expect to have
	// merged stores.
	// LET'S CHECK THAT LATER
	time.Sleep(100 * time.Millisecond) // And remove this!

	out = map[string]*state.Store{}
	var errs []string
	for _, squashable := range s.squashables {
		func() {
			squashable.RLock()         // REVISE the use of this lock
			defer squashable.RUnlock() // REVISE

			// the second check was added to take care of the use-case of a subsequent execution of the same request
			// where the target wasn't "reached" because there were no ranges to be done
			if !squashable.targetReached && !squashable.IsEmpty() {
				errs = append(errs, fmt.Sprintf("module %q: target %d not reached (ranges left: %s, next expected: %d)", squashable.name, s.targetExclusiveBlock, squashable.ranges, squashable.nextExpectedStartBlock))
			}
			if !squashable.IsEmpty() {
				errs = append(errs, fmt.Sprintf("module %q: missing ranges %s", squashable.name, squashable.ranges))
			}
			out[squashable.store.Name] = squashable.store
		}()
	}
	if len(errs) != 0 {
		return nil, fmt.Errorf("%d errors: %s", len(errs), strings.Join(errs, "; "))
	}
	return out, nil
}
