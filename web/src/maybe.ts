export default class Maybe<T> {
    private static readonly _EMPTY = new Maybe<never>(undefined);
    private readonly _value: T | undefined;

    private constructor(value: T | undefined) {
        this._value = value;
    }

    static of<T>(value: T | null | undefined): Maybe<NonNullable<T>> {
        return value == null
            ? Maybe.empty<NonNullable<T>>()
            : new Maybe<NonNullable<T>>(value as NonNullable<T>);
    }

    static empty<T = never>(): Maybe<T> {
        return Maybe._EMPTY as Maybe<T>;
    }

    isEmpty(): boolean {
        return this._value === undefined;
    }

    get(): T | undefined {
        return this.isEmpty()
            ? undefined
            : (this._value as T);
    }

    orElseGet(defaultValue: T): T {
        return this.isEmpty()
            ? defaultValue
            : (this._value as T);
    }

    map<U>(transformer: (value: T) => U): Maybe<NonNullable<U>> {
        return this.isEmpty()
            ? Maybe.empty<NonNullable<U>>()
            : Maybe.of<U>(transformer(this._value as T));
    }

    flatMap<U>(transformer: (value: T) => Maybe<U>): Maybe<U> {
        return this.isEmpty()
            ? Maybe.empty<U>()
            : transformer(this._value as T);
    }

    filter<S extends T>(predicate: (value: T) => value is S): Maybe<S>;
    filter(predicate: (value: T) => boolean): Maybe<T>;
    filter(predicate: (value: T) => boolean): Maybe<T> {
        if (this.isEmpty() || !predicate(this._value as T)) {
            return Maybe.empty<T>();
        }

        return this;
    }

    equals(other: Maybe<unknown>): boolean {
        return this.get() === other.get();
    }

    ifPresent(consumer: (value: T) => void): void {
        if (!this.isEmpty()) {
            consumer(this._value as T);
        }
    }
}